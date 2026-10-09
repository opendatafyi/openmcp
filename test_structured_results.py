"""Reliability and structured-output tests for OpenMCP's core tools."""

import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import mcp_server


EXPECTED_FIELDS = {
    "datasets", "columns", "rows", "sources", "query", "warnings", "error"
}


class TestStructuredToolContract(unittest.TestCase):

    def setUp(self):
        os.environ.pop("OPENDATA_FYI_TELEMETRY_ENABLED", None)
        os.environ.pop("OPENMCP_TELEMETRY_ENABLED", None)

    def assert_contract(self, result, *, is_error=False):
        self.assertEqual(set(result.structuredContent), EXPECTED_FIELDS)
        self.assertEqual(result.isError, is_error)
        self.assertEqual(result.content[0].type, "text")
        self.assertTrue(result.content[0].text)

    def test_core_tools_publish_the_same_output_schema(self):
        for name in (
            "semantic_search_datasets",
            "get_dataset",
            "query_statcan_wds",
            "query_datastore",
            "query_remote_file",
        ):
            tool = mcp_server.mcp._tool_manager.get_tool(name)
            self.assertIsNotNone(tool.fn_metadata.output_schema)
            self.assertEqual(
                set(tool.fn_metadata.output_schema["properties"]),
                EXPECTED_FIELDS,
            )

    @patch("mcp_server.check_catalog_compatibility", return_value=None)
    @patch("mcp_server.get_by_ids")
    @patch("mcp_server._ckan_get")
    @patch("mcp_server.top_k")
    @patch("mcp_server.embed_texts")
    @patch("mcp_server.os.path.exists", return_value=True)
    def test_semantic_search_returns_datasets_sources_and_markdown(
        self, _exists, embed_texts, top_k, ckan_get, get_by_ids, _check_compat
    ):
        native_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        dataset_id = f"canada:{native_id}"
        resources = [{
            "id": "resource-1",
            "name": "Housing data",
            "format": "CSV",
            "url": "https://example.ca/housing.csv",
            "datastore_active": False,
        }]
        embed_texts.return_value = [[0.1] * 384]
        top_k.return_value = [{"id": dataset_id, "distance": 0.1}]
        def search_response(_action, source_id="canada", **_params):
            if source_id != "canada":
                return {"results": []}
            return {"results": [{
                "id": native_id,
                "title": "Housing prices",
                "notes": "Housing price data",
                "organization": {"title": "Statistics Canada"},
                "resources": resources,
            }]}

        ckan_get.side_effect = search_response
        get_by_ids.return_value = {
            dataset_id: {
                "id": dataset_id,
                "title": "Housing prices",
                "org": "Statistics Canada",
                "notes": "Housing price data",
                "topic": "Housing",
                "resources": resources,
                "metadata_modified": "2026-01-01",
                "distance": None,
            }
        }

        result = mcp_server.semantic_search_datasets("housing prices", limit=5)

        self.assert_contract(result)
        self.assertEqual(result.structuredContent["datasets"][0]["id"], dataset_id)
        self.assertEqual(result.structuredContent["sources"][0]["dataset_id"], dataset_id)
        self.assertIn("Housing prices", result.content[0].text)

    @patch("mcp_server._ckan_get")
    def test_get_dataset_returns_machine_readable_resources(self, ckan_get):
        native_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        dataset_id = f"canada:{native_id}"
        ckan_get.return_value = {
            "id": native_id,
            "title": "Population estimates",
            "notes": "Annual estimates",
            "organization": {"title": "Statistics Canada"},
            "metadata_modified": "2026-01-02",
            "resources": [{
                "id": "resource-1",
                "name": "Estimates",
                "format": "CSV",
                "url": "https://example.ca/population.csv",
                "datastore_active": True,
            }],
        }

        result = mcp_server.get_dataset(native_id)

        self.assert_contract(result)
        dataset = result.structuredContent["datasets"][0]
        self.assertTrue(dataset["resources"][0]["datastore_active"])
        self.assertEqual(dataset["resources"][0]["format"], "CSV")
        self.assertIn("Resources", result.content[0].text)

    def test_datastore_rejects_non_object_filters_with_recovery_guidance(self):
        result = mcp_server.query_datastore("resource-1", filters='["Alberta"]')

        self.assert_contract(result, is_error=True)
        self.assertEqual(result.structuredContent["error"]["code"], "InvalidFilters")
        self.assertTrue(result.structuredContent["error"]["recovery"])

    @patch("mcp_server._ckan_get")
    def test_datastore_caps_rows_and_preserves_dataset_citation(self, ckan_get):
        native_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        dataset_id = f"canada:{native_id}"

        def response(action, **params):
            if action == "datastore_search":
                self.assertEqual(params["limit"], mcp_server.MAX_RESULT_ROWS)
                return {
                    "total": 2,
                    "records": [
                        {"_id": 1, "city": "Calgary", "value": 10},
                        {"_id": 2, "city": "Edmonton", "value": 20},
                    ],
                }
            return {"package_id": native_id}

        ckan_get.side_effect = response
        result = mcp_server.query_datastore("resource-1", limit=1000)

        self.assert_contract(result)
        self.assertEqual(len(result.structuredContent["rows"]), 2)
        self.assertNotIn("_id", result.structuredContent["rows"][0])
        self.assertEqual(result.structuredContent["query"]["limit"], 100)
        self.assertEqual(result.structuredContent["sources"][0]["dataset_id"], dataset_id)

    @patch("mcp_server.run_query_with_timeout")
    def test_remote_query_returns_rows_columns_source_and_markdown(self, run_query):
        run_query.return_value = pd.DataFrame({
            "city": ["Calgary", "Edmonton"],
            "population": [1_300_000, 1_000_000],
        })
        url = "https://example.ca/population.csv"

        result = mcp_server.query_remote_file(
            url, "SELECT city, population FROM '{file}'"
        )

        self.assert_contract(result)
        self.assertEqual(len(result.structuredContent["rows"]), 2)
        self.assertEqual(result.structuredContent["columns"][0]["name"], "city")
        self.assertEqual(result.structuredContent["sources"][0]["url"], url)
        self.assertIn(url, result.content[0].text)

    def test_remote_query_rejects_multiple_statements(self):
        result = mcp_server.query_remote_file(
            "https://example.ca/data.csv",
            "SELECT * FROM '{file}'; DROP TABLE data",
        )

        self.assert_contract(result, is_error=True)
        self.assertEqual(
            result.structuredContent["error"]["code"],
            "UnsafeOrInvalidSQL",
        )


class TestDuckDBConnectionReliability(unittest.TestCase):

    @patch("mcp_server._new_duckdb_connection")
    def test_each_query_uses_and_closes_a_separate_connection(self, new_connection):
        first = Mock()
        second = Mock()
        first.execute.return_value.fetchdf.return_value = pd.DataFrame({"a": [1]})
        second.execute.return_value.fetchdf.return_value = pd.DataFrame({"a": [2]})
        new_connection.side_effect = [first, second]

        mcp_server.run_query_with_timeout("SELECT 1")
        mcp_server.run_query_with_timeout("SELECT 2")

        self.assertEqual(new_connection.call_count, 2)
        first.close.assert_called_once()
        second.close.assert_called_once()

    @patch("mcp_server._new_duckdb_connection")
    def test_timeout_interrupts_and_closes_the_query_connection(self, new_connection):
        interrupted = threading.Event()

        class SlowConnection:
            closed = False

            def execute(self, _sql):
                while not interrupted.is_set():
                    time.sleep(0.001)
                raise RuntimeError("interrupted")

            def interrupt(self):
                interrupted.set()

            def close(self):
                self.closed = True

        connection = SlowConnection()
        new_connection.return_value = connection

        with self.assertRaises(TimeoutError):
            mcp_server.run_query_with_timeout("SELECT slow()", timeout_sec=0.01)

        self.assertTrue(interrupted.is_set())
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
