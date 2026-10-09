"""Tests for source-qualified catalog ingestion and routing."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import duckdb
import requests

import mcp_server
from semantic import build_index
from semantic import store
from resource_policy import is_queryable_resource
from source_registry import qualify_dataset_id, split_dataset_id


class TestSourceRegistry(unittest.TestCase):

    def test_legacy_qualified_and_page_ids_are_parsed(self):
        self.assertEqual(split_dataset_id("legacy-id"), ("canada", "legacy-id"))
        self.assertEqual(
            split_dataset_id("alberta:provincial-id"),
            ("alberta", "provincial-id"),
        )
        self.assertEqual(
            split_dataset_id("https://data.ontario.ca/dataset/ontario-id"),
            ("ontario", "ontario-id"),
        )
        self.assertEqual(
            split_dataset_id(
                "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901"
            ),
            ("statcan", "17100009"),
        )
        self.assertEqual(
            qualify_dataset_id("canada", "legacy-id"), "canada:legacy-id"
        )

    def test_new_portals_url_parsing_and_registry(self):
        self.assertEqual(
            split_dataset_id("https://open.toronto.ca/dataset/daily-shelter-occupancy"),
            ("toronto", "daily-shelter-occupancy"),
        )
        self.assertEqual(
            split_dataset_id("https://www.donneesquebec.ca/recherche/dataset/arbres-remarquables"),
            ("quebec", "arbres-remarquables"),
        )
        self.assertEqual(
            split_dataset_id("https://catalogue.data.gov.bc.ca/dataset/bc-air-quality"),
            ("bc", "bc-air-quality"),
        )
        self.assertEqual(
            split_dataset_id("https://www.bankofcanada.ca/valet/series/FXUSDCAD"),
            ("boc", "FXUSDCAD"),
        )

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            split_dataset_id("unknown:dataset")


class TestCatalogMigration(unittest.TestCase):

    def test_legacy_federal_catalog_migrates_without_losing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "catalog.duckdb")
            connection = duckdb.connect(db_path)
            connection.execute("""
                CREATE TABLE datasets (
                    id VARCHAR PRIMARY KEY, title VARCHAR, org VARCHAR,
                    notes VARCHAR, topic VARCHAR, resources_json VARCHAR,
                    metadata_modified VARCHAR, embedding FLOAT[384]
                )
            """)
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?::FLOAT[384])",
                ["legacy-id", "Legacy", "Canada", "Notes", "Topic", "[]", "", [0.0] * 384],
            )
            connection.close()

            new_record = {
                "id": "statcan:17100009",
                "native_id": "test-record",
                "source_id": "statcan",
                "source_type": "statcan_wds",
                "title": "Statistics Canada table",
                "org": "Statistics Canada",
                "notes": "Notes",
                "topic": "Topic",
                "resources": [{"url": "https://example.ca/data.csv"}],
                "page_url": "https://example.ca/dataset",
                "metadata": {"catalog_field": "preserved"},
                "embedding": [0.0] * 384,
            }
            with patch.object(store, "DB_PATH", db_path):
                store.save_datasets([new_record])
                records = store.get_by_ids(["legacy-id", "statcan:17100009"])

            self.assertEqual(set(records), {"canada:legacy-id", "statcan:17100009"})
            self.assertEqual(records["canada:legacy-id"]["native_id"], "legacy-id")
            self.assertEqual(records["statcan:17100009"]["source_type"], "statcan_wds")
            self.assertEqual(
                records["statcan:17100009"]["metadata"]["catalog_field"],
                "preserved",
            )

    def test_completed_source_rebuild_prunes_stale_records(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "catalog.duckdb")
            records = []
            for native_id in ("keep", "remove"):
                records.append({
                    "id": f"alberta:{native_id}",
                    "native_id": native_id,
                    "source_id": "alberta",
                    "source_type": "ckan",
                    "title": native_id,
                    "org": "Alberta",
                    "notes": "",
                    "topic": "",
                    "resources": [{"url": f"https://example.ca/{native_id}.csv"}],
                    "page_url": f"https://open.alberta.ca/dataset/{native_id}",
                    "embedding": [0.0] * store.EMBED_DIM,
                })
            with patch.object(store, "DB_PATH", db_path):
                store.save_datasets(records)
                store.prune_source_records("alberta", ["alberta:keep"])
                remaining = store.get_by_ids(["alberta:keep", "alberta:remove"])

            self.assertEqual(set(remaining), {"alberta:keep"})


class TestQueryableResourcePolicy(unittest.TestCase):

    def test_only_resources_with_a_working_query_path_are_accepted(self):
        for fmt in ("CSV", "XLS", "XLSX", "PARQUET", "JSON", "GEOJSON", "PDF", "TXT", "ZIP"):
            self.assertTrue(is_queryable_resource({"format": fmt}))
        self.assertTrue(is_queryable_resource({"url": "https://example.ca/zones.geojson"}))
        for fmt in ("SHP", "WMS", "WFS", "HTML", "KML", "GPKG"):
            self.assertFalse(is_queryable_resource({"format": fmt}))
        self.assertTrue(is_queryable_resource({
            "format": "WMS", "datastore_active": True
        }))
        self.assertTrue(is_queryable_resource({
            "format": "ZIP", "name": "English CSV files",
            "url": "https://example.ca/data.zip",
        }))
        self.assertFalse(is_queryable_resource({
            "format": "ZIP", "name": "Provincial shapefile",
            "url": "https://example.ca/boundaries.zip",
        }))

    def test_ckan_normalizer_drops_unsupported_resources_and_datasets(self):
        raw = {
            "id": "dataset-id",
            "title": "Mixed resources",
            "resources": [
                {"id": "csv", "format": "CSV", "url": "https://example.ca/data.csv"},
                {"id": "wms", "format": "WMS", "url": "https://example.ca/wms"},
            ],
        }
        record = build_index.process_dataset_dict(raw, source_id="alberta")
        self.assertEqual([resource["id"] for resource in record["resources"]], ["csv"])

        raw["resources"] = [{"id": "html", "format": "HTML", "url": "https://example.ca"}]
        self.assertIsNone(build_index.process_dataset_dict(raw, source_id="alberta"))

    def test_ckan_normalizer_handles_bilingual_metadata_and_language_priority(self):
        raw = {
            "id": "bilingual-id",
            "title_translated": {"en": "Electric Vehicle Rebates", "fr": "Rabais pour véhicules électriques"},
            "notes_translated": {"en": "English summary of EV rebates.", "fr": "Sommaire français des rabais pour VE."},
            "organization": {
                "title_translated": {"en": "Transport Canada", "fr": "Transports Canada"}
            },
            "topic_category": "Transport",
            "keywords": ["EV", "Vehicles", "Véhicules"],
            "resources": [
                {"id": "csv1", "name": "EV Data", "format": "CSV", "url": "https://example.ca/ev.csv"}
            ],
        }
        record = build_index.process_dataset_dict(raw, source_id="canada")
        self.assertIsNotNone(record)
        self.assertEqual(record["title"], "Electric Vehicle Rebates")
        self.assertIn("Electric Vehicle Rebates / Rabais pour véhicules électriques", record["doc_text"])
        self.assertIn("English summary of EV rebates.", record["doc_text"])
        self.assertIn("Sommaire français des rabais pour VE.", record["doc_text"])
        self.assertIn("Véhicules", record["doc_text"])


class TestStatisticsCanadaCatalog(unittest.TestCase):

    def test_cube_normalization_preserves_complete_metadata(self):
        cube = {
            "productId": 17100009,
            "cansimId": "051-0005",
            "cubeTitleEn": "Population estimates, quarterly",
            "cubeTitleFr": "Estimations de la population, trimestrielles",
            "cubeStartDate": "1946-01-01T05:00:00Z",
            "cubeEndDate": "2026-04-01T04:00:00Z",
            "releaseTime": "2026-06-17T12:30:00Z",
            "archived": "2",
            "subjectCode": ["17"],
            "surveyCode": ["3601"],
            "frequencyCode": 9,
            "corrections": [],
            "issueDate": "2018-06-27T04:00:00Z",
            "dimensions": [{
                "dimensionNameEn": "Geography",
                "dimensionNameFr": "Géographie",
                "dimensionPositionId": 1,
                "hasUOM": True,
            }],
        }
        code_sets = {
            "subject": [{"subjectCode": "17", "subjectEn": "Population and demography"}],
            "survey": [{"surveyCode": "3601", "surveyEn": "Quarterly Demographic Estimates"}],
            "frequency": [{"frequencyCode": 9, "frequencyDescEn": "Quarterly"}],
        }

        record = build_index.process_statcan_cube(cube, code_sets)

        self.assertEqual(record["id"], "statcan:17100009")
        self.assertEqual(record["metadata"]["cubeTitleFr"], cube["cubeTitleFr"])
        self.assertEqual(record["metadata"]["dimensions"], cube["dimensions"])
        self.assertEqual(record["metadata"]["status"], "active")
        self.assertIn("Geography", record["doc_text"])
        self.assertEqual(
            record["resources"][0]["url"],
            "https://www150.statcan.gc.ca/n1/tbl/csv/17100009-eng.zip",
        )

    @patch("semantic.build_index.requests.get")
    def test_full_catalog_uses_two_bulk_wds_endpoints(self, get):
        inventory_response = Mock()
        inventory_response.json.return_value = [{
            "productId": 17100009,
            "cubeTitleEn": "Population estimates, quarterly",
            "archived": "2",
            "subjectCode": [],
            "surveyCode": [],
            "frequencyCode": 9,
            "dimensions": [],
        }]
        codes_response = Mock()
        codes_response.json.return_value = {
            "status": "SUCCESS",
            "object": {"subject": [], "survey": [], "frequency": []},
        }
        get.side_effect = [inventory_response, codes_response]

        records = build_index.fetch_statcan_catalog()

        self.assertEqual(len(records), 1)
        self.assertIn("getAllCubesList", get.call_args_list[0].args[0])
        self.assertIn("getCodeSets", get.call_args_list[1].args[0])
        inventory_response.raise_for_status.assert_called_once()
        codes_response.raise_for_status.assert_called_once()


class TestStatCanWDSTool(unittest.TestCase):

    @patch("mcp_server._dl_session.post")
    def test_cube_metadata_routes_to_allowlisted_post_and_bounds_response(self, post):
        response = Mock()
        response.json.return_value = [{
            "status": "SUCCESS",
            "object": {
                "productId": 17100009,
                "dimension": [{
                    "member": [
                        {"memberId": 1},
                        {"memberId": 2},
                        {"memberId": 3},
                    ]
                }],
            },
        }]
        post.return_value = response

        result = mcp_server.query_statcan_wds(
            "getCubeMetadata", product_id="statcan:17100009", max_items=2
        )

        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["query"]["http_method"], "POST")
        self.assertEqual(
            result.structuredContent["sources"][0]["dataset_id"],
            "statcan:17100009",
        )
        self.assertTrue(result.structuredContent["warnings"])
        post.assert_called_once_with(
            "https://www150.statcan.gc.ca/t1/wds/rest/getCubeMetadata",
            json=[{"productId": 17100009}],
            timeout=mcp_server.HTTP_TIMEOUT,
        )
        response.raise_for_status.assert_called_once()

    @patch("mcp_server._dl_session.get")
    def test_full_csv_download_routes_to_product_and_language_path(self, get):
        response = Mock()
        response.json.return_value = {
            "status": "SUCCESS",
            "object": "https://www150.statcan.gc.ca/n1/tbl/csv/17100009-fra.zip",
        }
        get.return_value = response

        result = mcp_server.query_statcan_wds(
            "getFullTableDownloadCSV", product_id="17100009", language="fr"
        )

        self.assertFalse(result.isError)
        get.assert_called_once_with(
            "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/17100009/fr",
            timeout=mcp_server.HTTP_TIMEOUT,
        )

    @patch("mcp_server._dl_session.post")
    @patch("mcp_server._dl_session.get")
    def test_unknown_method_is_rejected_without_network_access(self, get, post):
        result = mcp_server.query_statcan_wds("deleteEverything")

        self.assertTrue(result.isError)
        self.assertEqual(
            result.structuredContent["error"]["code"],
            "InvalidStatCanWDSRequest",
        )
        get.assert_not_called()
        post.assert_not_called()

    @patch("mcp_server._dl_session.post")
    def test_latest_vector_data_normalizes_v_prefixes(self, post):
        response = Mock()
        response.json.return_value = [{"status": "SUCCESS", "object": {"vectorId": 1}}]
        post.return_value = response

        result = mcp_server.query_statcan_wds(
            "getDataFromVectorsAndLatestNPeriods",
            vector_ids="V1, v2",
            latest_n=4,
        )

        self.assertFalse(result.isError)
        post.assert_called_once_with(
            "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods",
            json=[{"vectorId": 1, "latestN": 4}, {"vectorId": 2, "latestN": 4}],
            timeout=mcp_server.HTTP_TIMEOUT,
        )


class TestBoCValetTool(unittest.TestCase):

    @patch("mcp_server._dl_session.get")
    def test_observations_routes_multiple_series_and_bounds_response(self, get):
        response = Mock()
        response.json.return_value = {
            "terms": {"url": "https://www.bankofcanada.ca/terms/"},
            "seriesDetail": {
                "FXUSDCAD": {"label": "USD/CAD"},
                "FXEURCAD": {"label": "EUR/CAD"},
            },
            "observations": [
                {"d": "2026-01-01", "FXUSDCAD": {"v": "1.35"}},
                {"d": "2026-01-02", "FXUSDCAD": {"v": "1.36"}},
                {"d": "2026-01-03", "FXUSDCAD": {"v": "1.37"}},
            ],
        }
        get.return_value = response

        result = mcp_server.query_boc_valet(
            "observations", series_names="FXUSDCAD, FXEURCAD", recent=3, max_items=2
        )

        self.assertFalse(result.isError)
        self.assertEqual(
            result.structuredContent["sources"][0]["dataset_id"], "boc:FXUSDCAD"
        )
        self.assertTrue(result.structuredContent["warnings"])
        get.assert_called_once_with(
            "https://www.bankofcanada.ca/valet/observations/FXUSDCAD,FXEURCAD/json",
            params={"recent": 3},
            timeout=mcp_server.HTTP_TIMEOUT,
        )
        response.raise_for_status.assert_called_once()

    @patch("mcp_server._dl_session.get")
    def test_group_metadata_routes_to_group_endpoint(self, get):
        response = Mock()
        response.json.return_value = {
            "groupDetail": {"name": "FX_RATES_DAILY", "label": "Daily exchange rates"},
        }
        get.return_value = response

        result = mcp_server.query_boc_valet("group", group_name="FX_RATES_DAILY")

        self.assertFalse(result.isError)
        self.assertEqual(
            result.structuredContent["sources"][0]["dataset_id"], "boc:FX_RATES_DAILY"
        )
        get.assert_called_once_with(
            "https://www.bankofcanada.ca/valet/groups/FX_RATES_DAILY/json",
            params={},
            timeout=mcp_server.HTTP_TIMEOUT,
        )

    @patch("mcp_server._dl_session.get")
    def test_series_list_does_not_require_names(self, get):
        response = Mock()
        response.json.return_value = {"series": {"FXUSDCAD": {"label": "USD/CAD"}}}
        get.return_value = response

        result = mcp_server.query_boc_valet("series_list")

        self.assertFalse(result.isError)
        get.assert_called_once_with(
            "https://www.bankofcanada.ca/valet/lists/series/json",
            params={},
            timeout=mcp_server.HTTP_TIMEOUT,
        )

    @patch("mcp_server._dl_session.get")
    def test_series_list_is_bounded_by_max_items_not_just_truncated_strings(self, get):
        # Valet's catalogue is a dict keyed by series name, not a list — the
        # generic list truncator can't see it, so this must be bounded
        # explicitly or max_items is silently ignored.
        response = Mock()
        response.json.return_value = {
            "series": {
                f"SERIES_{i}": {"label": f"Series {i}"} for i in range(10)
            }
        }
        get.return_value = response

        result = mcp_server.query_boc_valet("series_list", max_items=3)

        self.assertFalse(result.isError)
        self.assertEqual(len(result.structuredContent["rows"]), 3)
        self.assertEqual(result.structuredContent["rows"][0]["name"], "SERIES_0")
        self.assertTrue(result.structuredContent["warnings"])

    @patch("mcp_server._dl_session.get")
    def test_unknown_resource_is_rejected_without_network_access(self, get):
        result = mcp_server.query_boc_valet("deleteEverything")

        self.assertTrue(result.isError)
        self.assertEqual(
            result.structuredContent["error"]["code"], "InvalidBoCValetRequest"
        )
        get.assert_not_called()

    @patch("mcp_server._dl_session.get")
    def test_unknown_series_name_is_not_found_not_retryable(self, get):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(response=Mock(status_code=404))
        get.return_value = response

        result = mcp_server.query_boc_valet("series", series_names="NOT_A_REAL_SERIES")

        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["error"]["code"], "BoCValetNotFound")
        self.assertFalse(result.structuredContent["error"]["retryable"])

    @patch("mcp_server._dl_session.get")
    def test_recent_and_date_range_are_mutually_exclusive(self, get):
        result = mcp_server.query_boc_valet(
            "observations",
            series_names="FXUSDCAD",
            recent=5,
            start_date="2026-01-01",
        )

        self.assertTrue(result.isError)
        self.assertEqual(
            result.structuredContent["error"]["code"], "InvalidBoCValetRequest"
        )
        get.assert_not_called()


class TestSourceRouting(unittest.TestCase):

    def test_boc_dataset_id_redirects_to_query_boc_valet(self):
        result = mcp_server.get_dataset("boc:FXUSDCAD")

        self.assertTrue(result.isError)
        self.assertEqual(
            result.structuredContent["error"]["code"], "UnsupportedDatasetSource"
        )
        self.assertIn("query_boc_valet", result.structuredContent["error"]["recovery"])

    @patch("mcp_server._ckan_get")
    def test_alberta_dataset_routes_to_alberta_and_keeps_provenance(self, ckan_get):
        ckan_get.return_value = {
            "id": "provincial-id",
            "title": "Alberta dataset",
            "organization": {"title": "Government of Alberta"},
            "notes": "Description",
            "resources": [{
                "id": "resource-id",
                "name": "Data",
                "format": "CSV",
                "url": "https://example.ca/alberta.csv",
                "datastore_active": False,
            }],
        }

        result = mcp_server.get_dataset("alberta:provincial-id")

        ckan_get.assert_called_once_with(
            "package_show", source_id="alberta", id="provincial-id"
        )
        dataset = result.structuredContent["datasets"][0]
        self.assertEqual(dataset["id"], "alberta:provincial-id")
        self.assertEqual(dataset["source_id"], "alberta")
        self.assertEqual(
            result.structuredContent["sources"][0]["url"],
            "https://open.alberta.ca/dataset/provincial-id",
        )

    @patch("mcp_server._ckan_get")
    def test_datastore_and_citation_route_to_owning_portal(self, ckan_get):
        def response(action, source_id="canada", **_params):
            self.assertEqual(source_id, "ontario")
            if action == "datastore_search":
                return {"total": 1, "records": [{"value": 7}]}
            return {"package_id": "ontario-id"}

        ckan_get.side_effect = response
        result = mcp_server.query_datastore(
            "resource-id", source_id="ontario"
        )

        self.assertFalse(result.isError)
        self.assertEqual(
            result.structuredContent["sources"][0]["dataset_id"],
            "ontario:ontario-id",
        )
        self.assertEqual(
            result.structuredContent["sources"][0]["url"],
            "https://data.ontario.ca/dataset/ontario-id",
        )

    @patch("mcp_server.get_by_ids")
    @patch("mcp_server._ckan_get")
    def test_statcan_dataset_is_read_from_local_wds_index(self, ckan_get, get_by_ids):
        get_by_ids.return_value = {
            "statcan:17100009": {
                "id": "statcan:17100009",
                "native_id": "17100009",
                "source_id": "statcan",
                "source_type": "statcan_wds",
                "title": "Population estimates, quarterly",
                "org": "Statistics Canada",
                "notes": "Complete table metadata",
                "page_url": "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901",
                "metadata": {"frequencyLabelEn": "Quarterly"},
                "resources": [{
                    "id": "statcan-17100009-eng",
                    "name": "Full table",
                    "format": "ZIP",
                    "url": "https://www150.statcan.gc.ca/n1/tbl/csv/17100009-eng.zip",
                    "datastore_active": False,
                }],
            }
        }

        result = mcp_server.get_dataset("statcan:17100009")

        ckan_get.assert_not_called()
        self.assertEqual(
            result.structuredContent["datasets"][0]["metadata"]["frequencyLabelEn"],
            "Quarterly",
        )

    @patch("mcp_server.get_by_ids")
    @patch("mcp_server.check_catalog_compatibility", return_value=None)
    @patch("mcp_server._ckan_get")
    @patch("mcp_server.top_k", return_value=[])
    @patch("mcp_server.embed_texts", return_value=[[0.1] * store.EMBED_DIM])
    @patch("mcp_server.os.path.exists", return_value=True)
    def test_portal_failure_is_isolated_during_semantic_search(
        self, _exists, _embed, _top_k, ckan_get, _compat, get_by_ids
    ):
        def response(_action, source_id="canada", **_params):
            if source_id == "alberta":
                raise ConnectionError("unavailable")
            if source_id == "canada":
                return {"results": [{
                    "id": "federal-id",
                    "title": "Federal result",
                    "organization": {"title": "Government of Canada"},
                    "resources": [{
                        "id": "resource-id", "name": "Data", "format": "CSV",
                        "url": "https://example.ca/federal.csv",
                    }],
                }]}
            return {"results": []}

        ckan_get.side_effect = response
        get_by_ids.return_value = {}

        result = mcp_server.semantic_search_datasets("federal data")

        self.assertFalse(result.isError)
        self.assertEqual(
            result.structuredContent["datasets"][0]["id"], "canada:federal-id"
        )
        self.assertTrue(any("Alberta" in warning for warning in result.structuredContent["warnings"]))
    
    @patch("mcp_server.os.path.exists", return_value=True)
    @patch("mcp_server.check_catalog_compatibility")
    def test_catalog_model_mismatch_returns_error_result(self, check_compat, _exists):
        check_compat.return_value = {
            "expected_model": "google/embeddinggemma-2",
            "expected_dim": 768,
            "catalog_model": "BAAI/bge-small-en-v1.5",
            "catalog_dim": 384,
        }
        result = mcp_server.semantic_search_datasets("test query")
        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["error"]["code"], "CatalogModelMismatch")
        self.assertIn("Catalog embedding model mismatch", result.structuredContent["error"]["message"])
        self.assertIn("Rebuild catalog.duckdb", result.structuredContent["error"]["recovery"])

    @patch("mcp_server.get_by_ids")
    @patch("mcp_server.check_catalog_compatibility", return_value=None)
    @patch("mcp_server._ckan_get", return_value={"results": []})
    @patch("mcp_server.bm25_search", return_value=[])
    @patch("mcp_server.top_k", return_value=[])
    @patch("mcp_server.top_k_assets")
    @patch("mcp_server.embed_texts", return_value=[[0.1] * store.EMBED_DIM])
    @patch("mcp_server.os.path.exists", return_value=True)
    def test_child_asset_match_annotates_matched_via(
        self, _exists, _embed, top_k_assets_mock, _top_k, _bm25, _ckan, _compat, get_by_ids
    ):
        top_k_assets_mock.return_value = [{
            "asset_id": "canada:test-id:schema:res1",
            "dataset_id": "canada:test-id",
            "kind": "column_schema",
            "title": "CSV Columns: Table 1",
            "metadata": {"columns": ["TOTAL_EXPENDITURE", "YEAR", "REGION"]},
        }]
        get_by_ids.return_value = {
            "canada:test-id": {
                "id": "canada:test-id",
                "title": "Municipal Budget Dataset",
                "org": "Treasury Board",
                "resources": [],
            }
        }
        result = mcp_server.semantic_search_datasets("total expenditure by municipality")
        self.assertFalse(result.isError)
        datasets = result.structuredContent["datasets"]
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["id"], "canada:test-id")
        self.assertIn("Column schema", datasets[0]["matched_via"])
        self.assertIn("TOTAL_EXPENDITURE", datasets[0]["matched_via"])


class TestEnrichment(unittest.TestCase):

    def test_enrich_dataset_extracts_schema_and_pdf_assets(self):
        from semantic import enrich
        ds = {
            "id": "canada:test-enrich",
            "source_id": "canada",
            "resources": [
                {
                    "id": "res-ds",
                    "name": "Datastore Table",
                    "format": "CSV",
                    "url": "https://example.ca/data.csv",
                    "datastore_active": True,
                },
                {
                    "id": "res-pdf",
                    "name": "Methodology Guide",
                    "format": "PDF",
                    "url": "https://example.ca/guide.pdf",
                    "datastore_active": False,
                }
            ]
        }
        with patch.object(enrich, "extract_datastore_schema", return_value=["ID", "POPULATION", "CITY"]), \
             patch.object(enrich, "extract_pdf_text", return_value="This document outlines census methodology."):
            assets = enrich.enrich_dataset(ds)
            self.assertEqual(len(assets), 2)
            self.assertEqual(assets[0]["kind"], "column_schema")
            self.assertIn("POPULATION", assets[0]["text"])
            self.assertEqual(assets[1]["kind"], "pdf_text")
            self.assertIn("census methodology", assets[1]["text"])


if __name__ == "__main__":
    unittest.main()

