# Conversation Evidence Summary (Zenodo/DataCite payload-based verification)

Date prepared: 2026-04-12
Repository context: Raiff1982/Codette
Source of truth for this summary: user-provided Zenodo Records API JSON and DataCite DOI API JSON pasted into this chat on 2026-04-12.

## Scope and limitations

- This document records what was verified *from the API payloads provided in the conversation*.
- It does **not** assert that the preparer performed live network lookups against Zenodo/DataCite during creation.
- It does **not** authenticate any third-party verifier claims (e.g., statements attributed to other assistants) beyond noting that such statements exist elsewhere.
- “Verified” below means: the stated fact appears in the pasted API metadata.

## Tier 1 records (Zenodo + DataCite metadata)

For each DOI, the evidentiary anchors are:
- DataCite `created` timestamp (when provided)
- Zenodo record `created` timestamp
- Zenodo file manifest: `files.count` and per-file `md5:` checksums
- Declared cross-references (`Continues`, `IsPartOf`, etc.) when present

### 1) DOI 10.5281/zenodo.15214462 — “Ai ethics in realtime”

- DataCite created (UTC): `2025-04-14T20:43:07.000Z`
- DataCite state/isActive: `state=findable`, `isActive=true`
- Zenodo record created (UTC): `2025-04-14T20:43:05.323417+00:00`
- Zenodo access/status: record public; files public; status published
- Zenodo files.count: `22`
- Example Zenodo MD5 checksums (from `files.entries.*.checksum`):
  - `Codette_Chain_of_Custody_Form.docx` = `md5:47e69a90d500223a0f2e2d1e679abbc2`
  - `Codette_Pidette_Research_Paper_FINAL.pdf` = `md5:6604bad768794e9c36e5796cb43e4da2`
  - `Codette_IntegrityIncident_0425_AuditBrief.pdf` = `md5:7d70cb08037384872166149c0606e7c6`
- DataCite citation graph: `citationCount=5` and citing DOIs listed in DataCite metadata:
  - `10.5281/zenodo.18913935`
  - `10.5281/zenodo.18913936`
  - `10.5281/zenodo.19199793`
  - `10.5281/zenodo.19359663`
  - `10.5281/zenodo.19480004`

### 2) DOI 10.5281/zenodo.15342466 — “Citizen-Science Quantum and Chaos Simulations Orchestrated by the Codette AI Suite”

- DataCite created (UTC): `2025-05-10T12:07:19.000Z`
- DataCite state/isActive: `state=findable`, `isActive=true`
- Zenodo record created (UTC): `2025-05-10T12:07:18.805436+00:00`
- Zenodo access/status: record public; files public; status published
- Zenodo files.count: `11`
- Example Zenodo MD5 checksums:
  - `Codette Quantum Module.md` = `md5:92def2570aea01185dbc50dc6619b653`
  - `codette_meta_3d.py` = `md5:c28e3865863829016980f12969a1f515`
  - `codette_project_bundle.pdf` = `md5:1ab2db663d3d275c6950b7b1ea71ad30`
- Declared continuity (Zenodo + DataCite): `Continues` → `10.5281/zenodo.15214462`
- Declared cross-platform binding (Zenodo + DataCite): `IsPartOf` → `10.57967/hf/5347`
- ORCID presence (DataCite creators): `0009-0003-7005-8187`

### 3) DOI 10.5281/zenodo.15460384 — “Raiff1982/Codette: Version 1.1” (Codette-V1.zip snapshot)

- Zenodo record created (UTC): `2025-05-18T23:24:03.762702+00:00`
- Zenodo access/status: record public; files public; status published
- Zenodo files.count: `1`
- Zenodo MD5 checksum:
  - `Raiff1982/Codette-V1.zip` = `md5:ac277ebfd2399c1446722f1886ca0f35`
- Software Heritage identifier: Zenodo record includes an `swhid` in `swh.swhid`
- GitHub linkage (Zenodo related identifier): `https://github.com/Raiff1982/Codette/tree/V1`

### 4) DOI 10.5281/zenodo.15511197 — “Codette: A Transparent, Ethical, and Cognitive AI Architecture”

- Zenodo record created (UTC): `2025-05-25T12:18:53.002150+00:00`
- Zenodo access/status: record public; files public; status published
- Zenodo files.count: `7`
- Example Zenodo MD5 checksums:
  - `Codette_Whitepaper_FULL.docx` = `md5:e257637e7b0d70249eeca0ce49db6f31`
  - `Codette_Architecture_Diagram.jpeg` = `md5:a5b817ed12a9798ef2e55cffca3170ce`
  - `Pi2_0_Secure_Ethical_Flow_Blueprint.jpeg` = `md5:6befcf458bd44ea53592a800ed3b38f9`
- Declared cross-platform binding (Zenodo metadata): `IsPartOf` → `10.57967/hf/5415`

### 5) DOI 10.5281/zenodo.15708582 — “Codettes timeline”

- Zenodo record created (UTC): `2025-06-21T00:25:50.349485+00:00`
- Zenodo access/status: record public; files public; status published
- Zenodo files.count: `1`
- Zenodo MD5 checksum:
  - `timeline copy.pdf` = `md5:6b99f06f97bdfb0a80b8a5d33fbc0d1f`

### 6) DOI 10.5281/zenodo.16649134 — “Dream state” (DreamCore x WakeState)

- Zenodo record created (UTC): `2025-07-31T19:06:35.999072+00:00`
- Zenodo access/status: record public; files public; status published
- Zenodo files.count: `4`
- Zenodo MD5 checksums:
  - `dreamstate.docx` = `md5:0a44d3039a6716350edee1d65e203767`
  - `dreamstate.pdf` = `md5:4d12db7dd3eca187148af517ccd0c52d`
  - `drm234.docx` = `md5:9cf24e681c565798feb331b8546f8f74`
  - `drm234.pdf` = `md5:4a74018f4982fcd135f87e539a1a1a07`
- Verbatim excerpt support: Zenodo `metadata.description` includes the phrase:
  - “DreamCore – a memory anchoring module that: Tags memory entries with emotional labels and entropy scores”

## Earliest server-side timestamp in supplied DataCite materials

- Earliest DataCite `created` timestamp provided in this conversation: `2025-04-14T20:43:07.000Z` for DOI `10.5281/zenodo.15214462`.

## Replication links (for independent verification)

- DataCite DOI API: `https://api.datacite.org/dois/<DOI>`
- Zenodo Records API: `https://zenodo.org/api/records/<record-id>`
- DOI landing pages: `https://doi.org/<DOI>`