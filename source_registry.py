"""Authoritative data-source registry and dataset identifier helpers."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceConfig:
    id: str
    name: str
    source_type: str
    api_base: Optional[str]
    page_template: str
    language_priority: Tuple[str, ...]

    def page_url(self, native_id: str) -> str:
        return self.page_template.format(native_id=native_id)


SOURCE_REGISTRY: Dict[str, SourceConfig] = {
    "canada": SourceConfig(
        id="canada",
        name="Government of Canada Open Data",
        source_type="ckan",
        api_base="https://open.canada.ca/data/en/api/3/action",
        page_template="https://open.canada.ca/data/en/dataset/{native_id}",
        language_priority=("en", "fr"),
    ),
    "alberta": SourceConfig(
        id="alberta",
        name="Government of Alberta Open Data",
        source_type="ckan",
        api_base="https://open.alberta.ca/api/3/action",
        page_template="https://open.alberta.ca/dataset/{native_id}",
        language_priority=("en",),
    ),
    "ontario": SourceConfig(
        id="ontario",
        name="Government of Ontario Data Catalogue",
        source_type="ckan",
        api_base="https://data.ontario.ca/api/3/action",
        page_template="https://data.ontario.ca/dataset/{native_id}",
        language_priority=("en", "fr"),
    ),
    "statcan": SourceConfig(
        id="statcan",
        name="Statistics Canada Data Tables",
        source_type="statcan_wds",
        api_base="https://www150.statcan.gc.ca/t1/wds/rest",
        page_template="https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={native_id}01",
        language_priority=("en", "fr"),
    ),
    "bc": SourceConfig(
        id="bc",
        name="Government of British Columbia Data Catalogue",
        source_type="ckan",
        api_base="https://catalogue.data.gov.bc.ca/api/3/action",
        page_template="https://catalogue.data.gov.bc.ca/dataset/{native_id}",
        language_priority=("en",),
    ),
    "toronto": SourceConfig(
        id="toronto",
        name="City of Toronto Open Data",
        source_type="ckan",
        api_base="https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action",
        page_template="https://open.toronto.ca/dataset/{native_id}",
        language_priority=("en",),
    ),
    "quebec": SourceConfig(
        id="quebec",
        name="Données Québec",
        source_type="ckan",
        api_base="https://www.donneesquebec.ca/recherche/api/3/action",
        page_template="https://www.donneesquebec.ca/recherche/dataset/{native_id}",
        language_priority=("fr", "en"),
    ),
    "boc": SourceConfig(
        id="boc",
        name="Bank of Canada Valet API",
        source_type="boc_valet",
        api_base="https://www.bankofcanada.ca/valet",
        page_template="https://www.bankofcanada.ca/valet/series/{native_id}",
        language_priority=("en",),
    ),
}

CKAN_SOURCE_IDS = tuple(
    source_id
    for source_id, config in SOURCE_REGISTRY.items()
    if config.source_type == "ckan"
)
INDEX_SOURCE_IDS = (*CKAN_SOURCE_IDS, "statcan")


def get_source(source_id: str) -> SourceConfig:
    """Return a registered source or raise a useful validation error."""
    normalized = (source_id or "").strip().lower()
    if normalized not in SOURCE_REGISTRY:
        allowed = ", ".join(SOURCE_REGISTRY)
        raise ValueError(f"Unknown source '{source_id}'. Expected one of: {allowed}.")
    return SOURCE_REGISTRY[normalized]


def qualify_dataset_id(source_id: str, native_id: str) -> str:
    """Build the collision-safe catalog identifier used in DuckDB."""
    source = get_source(source_id)
    native = (native_id or "").strip()
    if not native:
        raise ValueError("A dataset ID is required.")
    if native.startswith(f"{source.id}:"):
        return native
    return f"{source.id}:{native}"


def split_dataset_id(dataset_id: str, default_source: str = "canada") -> Tuple[str, str]:
    """Parse a qualified ID, legacy federal ID, or known dataset page URL."""
    value = (dataset_id or "").strip()
    if not value:
        raise ValueError("A dataset ID is required.")

    if "://" in value:
        parsed = urlparse(value)
        host = parsed.netloc.lower().split(":", 1)[0]
        if host in ("www150.statcan.gc.ca", "www.statcan.gc.ca"):
            query = parsed.query.split("&")
            pid_value = next(
                (part.split("=", 1)[1] for part in query if part.startswith("pid=")),
                "",
            )
            if len(pid_value) == 10 and pid_value.endswith("01"):
                return "statcan", pid_value[:-2]
            raise ValueError(f"Could not find a Statistics Canada PID in URL: {value}")
        if host in ("www.bankofcanada.ca", "bankofcanada.ca"):
            parts = [part for part in parsed.path.split("/") if part]
            if parts:
                return "boc", parts[-1]
            raise ValueError(f"Could not find a Bank of Canada series or group name in URL: {value}")
        parts = [part for part in parsed.path.split("/") if part]
        try:
            native = parts[parts.index("dataset") + 1]
        except (ValueError, IndexError):
            raise ValueError(f"Could not find a dataset ID in URL: {value}") from None
        if host == "open.canada.ca":
            return "canada", native
        if host == "open.alberta.ca":
            return "alberta", native
        if host == "data.ontario.ca":
            return "ontario", native
        if host in ("catalogue.data.gov.bc.ca", "data.gov.bc.ca"):
            return "bc", native
        if host == "open.toronto.ca":
            return "toronto", native
        if host in ("www.donneesquebec.ca", "donneesquebec.ca"):
            return "quebec", native
        raise ValueError(f"Dataset URL host is not registered: {host}")

    if ":" in value:
        source_id, native_id = value.split(":", 1)
        get_source(source_id)
        if not native_id:
            raise ValueError("A native dataset ID is required after the source prefix.")
        return source_id.lower(), native_id

    get_source(default_source)
    return default_source, value


def page_url(source_id: str, native_id: str) -> str:
    """Return the public dataset page for a source with generated page URLs."""
    config = get_source(source_id)
    return config.page_url(native_id)
