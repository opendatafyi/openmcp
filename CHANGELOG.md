# Changelog

All notable changes to opendata.fyi OpenMCP are documented here.

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `query_boc_valet`, a validated read-only gateway to the Bank of Canada
  Valet API (series, groups, and observations) for interest rates, exchange
  rates, and other financial series. No API key required. Live-query only —
  not part of the semantic index or `get_dataset`.
- Tests for the Valet request builder, mutually-exclusive range parameters,
  and response bounding.

## [1.1.0] - 2026-08-03

### Added

- Alberta Open Data and Ontario Open Data as first-class CKAN sources.
- The complete Statistics Canada WDS inventory, including active and archived
  tables, metadata, vectors, coordinates, time series, and bulk CSV access.
- `query_statcan_wds`, a validated read-only gateway to the documented WDS
  methods and code sets.
- Source-qualified dataset IDs such as `alberta:…`, `ontario:…`, and
  `statcan:17100009`.
- Consistent MCP `structuredContent` for discovery and core query tools, with
  datasets, columns, rows, sources, query details, warnings, and stable errors.
- Tests for source routing, catalogue migration, WDS requests, resource policy,
  structured results, DuckDB connection handling, and telemetry privacy.

### Changed

- Expanded the downloadable semantic index from one federal catalogue to
  74,728 queryable datasets and statistical tables across four official
  sources.
- Hybrid discovery now combines the shared local semantic index with concurrent
  live keyword searches of the federal, Alberta, and Ontario CKAN catalogues.
- Dataset and resource requests now preserve their owning source and official
  citation page.
- Index rebuilds prune stale records per source and incremental refreshes can
  target selected sources.
- Remote file and datastore queries return bounded, machine-readable results
  while retaining Markdown compatibility.
- Anonymous telemetry remains off by default and now records only normalized
  errors, latency, the server version, and public dataset IDs when explicitly
  enabled.

### Compatibility

- Existing bare Government of Canada dataset IDs remain supported.
- Existing MCP clients can continue using Markdown tool output while clients
  that support structured results receive the richer response schema.

## [1.0.0] - 2026-07-23

### Added

- Initial public release with Government of Canada catalogue discovery,
  semantic search, CKAN datastore access, and remote CSV, Parquet, JSON, Excel,
  ZIP, text, and PDF querying.
