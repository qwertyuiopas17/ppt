```
  _   _      _   ____             _   _            _
 | \ | | ___| |_/ ___|  ___ _ __ | |_(_)_ __   ___| |
 |  \| |/ _ \ __\___ \ / _ \ '_ \| __| | '_ \ / _ \ |
 | |\  |  __/ |_ ___) |  __/ | | | |_| | | | |  __/ |
 |_| \_|\___|\__|____/ \___|_| |_|\__|_|_| |_|\___|_|

 Passive. Offline. Correlated.
 A machine learning intrusion detection sensor for unidirectional IP traffic.
```

NetSentinel is a passive, offline network intrusion detection system (NIDS). It ingests
network traffic from a PCAP file or a mirror/TAP interface, extracts behavioral metadata,
and routes it through six specialized detection models. Alerts stream to a browser
dashboard in real time over a WebSocket. The system inspects no payloads, sits inline with
nothing, and sends no data off the monitored network.

The novelty is not any single model. DGA classification, volumetric DDoS detection, and
anomaly-based exfiltration detection are established techniques. The contribution is the
integration: fusing six independent threat signals into correlated attack chains on one
passive sensor that runs entirely on local hardware, so that a DGA lookup, a C2 beacon, and
a DNS tunnel from the same internal host read as a single kill chain rather than three
disconnected alerts.

---

## Status at a glance

| Property | State |
|---|---|
| Deployment posture | Passive sensor. Detection and alerting only. No block, quarantine, or mitigation. |
| Models wired and routed | 6 of 6 (DDoS, Port Scan, DGA, Exfiltration, C2 Beacon, Encrypted Traffic) |
| Inference runtime | ONNX Runtime + scikit-learn, CPU only, no GPU required |
| External dependencies at runtime | None. No cloud, no API calls, no telemetry. Suitable for air-gapped networks. |
| Payload inspection | None. Headers, flow statistics, DNS strings, and timing metadata only. |
| Model distribution | HuggingFace Hub with local caching, offline capable after first fetch |
| Forensic integrity | Phase 1 complete. 7 of 9 claims functional (DSSE receipts, replay, Merkle batching, Ed25519 ledger). Phase 2–3 in progress. |

---

## Table of contents

1. [Design principles](#design-principles)
2. [System architecture](#system-architecture)
3. [Detection models](#detection-models)
4. [Extraction pipeline](#extraction-pipeline)
5. [Validated results](#validated-results)
6. [Dashboard](#dashboard)
7. [How NetSentinel compares](#how-netsentinel-compares)
8. [Technology stack](#technology-stack)
9. [Installation](#installation)
10. [Running the system](#running-the-system)
11. [API reference](#api-reference)
12. [Configuration](#configuration)
13. [Project structure](#project-structure)
14. [🔒 Forensic integrity layer](#-forensic-integrity-layer-experimental)
15. [Known limitations](#known-limitations)
16. [Roadmap](#roadmap)
17. [License](#license)

---

## Design principles

Three constraints shape every part of the system.

**Passive observation.** NetSentinel operates on a copy of network traffic taken from a
mirror port or TAP. It is not inline, does not modify packets, and cannot cause network
disruption. Its output is detection and evidence. Response decisions stay with the analyst.

**Offline operation.** All six models run locally through ONNX Runtime on commodity CPUs.
There are no cloud dependencies and no outbound telemetry. This is a deliberate design
target for environments where cloud-backed security tooling is not permitted: government
networks, defence installations, and SCADA/ICS infrastructure.

**Metadata only.** NetSentinel never reads packet payloads. It works from headers, flow
statistics, DNS query strings, and timing. This allows detection on encrypted traffic,
including TLS 1.3, without decryption, and keeps the system privacy-preserving by design.

Every model is exported to ONNX, giving a single inference interface across three very
different architectures (gradient-boosted trees, recurrent networks, and a transformer).

---

## System architecture

```
                        +--------------------------------------+
   PCAP file  ------>   |            EXTRACTION LAYER           |
   Live TAP   ------>   |                                      |
                        |  Phase 1: CICFlowMeter               |
                        |    59 flow-statistic features        |
                        |    tag = "cicflowmeter"              |
                        |                                      |
                        |  Phase 2: Custom Scapy extractor     |
                        |    DNS query strings                 |
                        |    24 DNS lexical features           |
                        |    100-flow session time-series      |
                        |    tag = "custom"                    |
                        +------------------+-------------------+
                                           |
                              type + source-tag routing
                                           |
        +------------------+---------------+----------------+------------------+
        |                  |               |                |                  |
        v                  v               v                v                  v
   +---------+       +-----------+   +-----------+    +-----------+     +-------------+
   |  DDoS   |       | Port Scan |   |    DGA    |    | Exfil VAE |     | C2 Beacon   |
   | XGBoost |       | SPSD RF + |   | CNN-BiLSTM|    | recon err |     | BiLSTM+FFT  |
   |         |       | fan-out   |   |  3-class  |    |           |     |             |
   +----+----+       +-----+-----+   +-----+-----+    +-----+-----+     +------+------+
        |                  |               |                |                  |
        |                  |         +-----------+          |                  |
        |                  |         | Encrypted |          |                  |
        |                  |         |    FT-    |          |                  |
        |                  |         | Transformer|         |                  |
        |                  |         +-----+-----+          |                  |
        +------------------+---------------+----------------+------------------+
                                           |
                                           v
                        +--------------------------------------+
                        |  ANALYZER  (correlation + evidence)  |
                        |  ALERT MANAGER  (severity, MITRE)    |
                        +------------------+-------------------+
                                           |
                                    WebSocket  ws://host:8000/ws
                                           |
                                           v
                        +--------------------------------------+
                        |  REACT DASHBOARD                     |
                        |  alert feed, 3D threat graph, FFT    |
                        |  spectrum, timeline, MITRE heatmap   |
                        +--------------------------------------+
```

The routing rule is strict and is the reason detection quality holds up on real captures:
CICFlowMeter events are only ever sent to models trained on CICFlowMeter data (DDoS, Port
Scan), and custom-extractor events are only ever sent to the models that need DNS strings
or session time-series (DGA, Exfiltration, C2, Encrypted Traffic). No model receives input
drawn from a distribution it was not trained on.

---

## Detection models

Each model targets one threat class. All six are wired into the registry and routed by the
analyzer.

| Model | Architecture | Training data | MITRE ATT&CK |
|---|---|---|---|
| DDoS | XGBoost binary classifier | CIC-DDoS2019 | T1498, T1499 |
| Port Scan | RandomForest SPSD (network-event aggregation) + fan-out backstop | CIDDS-001 + synthetic fast-scan augmentation | T1046 |
| DGA | Char-level CNN + BiLSTM, 3-class | DGArchive 2024, Tranco 1M, synthetic tunnels | T1568.002 |
| Exfiltration | Variational Autoencoder | CIC-Bell-DNS-EXF-2021 | T1048, T1071.004 |
| C2 Beacon | BiLSTM with FFT spectral features | Synthetic and labeled beacon series | T1071, T1573 |
| Encrypted Traffic | FT-Transformer, 14 classes | Consolidated VPN / non-VPN dataset | T1573, T1572 |

### DDoS

Each flow is summarized by CICFlowMeter into 59 statistical features covering traffic
volume, inter-arrival timing, TCP flag counts, and packet-size distributions. Automated
flood tools produce sub-millisecond, highly regular intervals and uniform small packets;
amplification attacks show small requests against large responses. An alert requires three
conditions in series to keep production false positives low:

1. Model confidence above 98 percent. This is deliberately conservative and trades a small
   detection delay for far fewer false alerts.
2. A rate guard confirming the flow actually exhibits high throughput (`Flow Packets/s > 100`
   or `Flow Bytes/s > 50000`), so slow flows that merely resemble DDoS in their ratios do
   not trigger.
3. An all-zero guard that rejects degenerate flows with no valid feature data.

Source-IP diversity per destination is tracked as supplementary botnet evidence.

### Port Scan

Port scanning is the reconnaissance phase of an intrusion. It is fundamentally a
**cross-flow phenomenon**: CICFlowMeter turns each probe into its own single-SYN flow with
no response, which is indistinguishable at the per-flow level from an ordinary unanswered
connection attempt. A per-flow classifier scores each probe as benign even during a real
999-port Nmap scan. The scan signal only exists when you aggregate across flows.

NetSentinel addresses this with a **network-event aggregation layer** based on the SPSD
algorithm from Hakem et al. ("Behavioral-Based Port Scan Detection Using Supervised Machine
Learning Algorithms"). CICFlowMeter flows are buffered in 60-second windows and grouped by
source IP. For each network event, six behavioral features are computed:

| Feature | Symbol | What it measures |
|---------|--------|------------------|
| ICMP Error | α₁ | ICMP unreachable responses (UDP scan indicator) |
| RST | α₂ | TCP RST responses (closed port indicator) |
| RWA | α₃ | Requests without answer (firewalled/dropped) |
| NEIP | α₄ | Probes to non-existent internal IPs |
| NETCP | α₅ | Probes to non-open ports on known hosts |
| Succession | α₆ | Cross-window persistence counter |

A **RandomForestClassifier** (20 trees, `max_depth=6`, `class_weight='balanced'`) trained
on CIDDS-001 real network data augmented with synthetic fast-scan events classifies these
network events. Detection is dual-mechanism:

- **SPSD model** — primary detector for concentrated scans. 82.6% detection rate on
  CIDDS-001 validation (1,878 / 2,274 scans), 7.4% false positive rate. Fast scans
  detected at 95.9% confidence.
- **Fan-out backstop** — time-independent cumulative threshold (≥100 distinct destination
  ports per source IP). Catches ultra-slow stealth scans that leave no per-window
  behavioral footprint. Validated on a 3.3-hour, 1000-port stealth scan from CIC-IDS-2017.

The NEIP and NETCP features require a `network_info.json` topology file documenting
internal subnets and known open ports. Without it, the system degrades gracefully to
RST/RWA/ICMP/succession features plus the fan-out backstop.

The old per-flow XGBoost model is preserved as an optional `ml_flow_score` field for
backward compatibility. It does not drive alerts.

### DGA

Domain Generation Algorithms let malware families such as Conficker, Necurs, Ramnit, and
Emotet rotate through hundreds of pseudo-random command-and-control domains per day, so
defenders cannot sinkhole a fixed address. The domain string is lowercased and encoded
character by character into a 128-element integer vector (vocabulary 40). A 1D CNN learns
local n-gram patterns that separate high-entropy generated strings from natural word
fragments, and a bidirectional LSTM captures domain-wide structure. The output is a 3-class
softmax: Benign, DGA, or DNS Tunnel.

The model is evaluated with family-wise holdout: 27 entire DGA families are held out of
training, which prevents the data leakage that random splits cause when near-identical
domains from one family land in both train and test. Under that honest split it reaches
0.9787 macro-F1, in the same 0.97 to 0.99 band reported by Cisco Umbrella, Elastic
(Endgame), and FANCI. The distinction is that NetSentinel classifies locally without
forwarding queries to an external resolver.

### Data Exfiltration over DNS

DNS tunneling encodes stolen data into DNS queries, using port 53 as a covert channel that
firewalls rarely block. Tools include dnscat2, iodine, dns2tcp, and Cobalt Strike DNS mode.
Detection is unsupervised. For each query, 24 lexical features are computed from the domain
string alone: Shannon and bigram entropy, character-composition ratios, length measures,
and structural counts. A Variational Autoencoder trained on benign DNS learns to reconstruct
normal query vectors; tunnel queries fall outside that distribution and reconstruct poorly,
producing high mean-squared error.

The separation is wide. Normal DNS lands at MSE 0.18 to 0.42; tunnel traffic at 0.99 to 140
(dns2tcp 0.99, iodine 1.44, dnscat2 61 to 140). The threshold sits at 0.70, with an
additional requirement that either the error exceeds 1.4 or the domain shows moderate
tunneling indicators (entropy above 4.0 with subdomain length above 20) before an alert
issues. Tested against the DNS-Tunnel-Datasets collection across seven tools and multiple
record types, the model reaches 100 percent true positive rate at 3.3 percent false
positive rate. An earlier build showed roughly 97.7 percent false positives because the
scaler was pickled with scikit-learn 1.6.1 and loaded under 1.3.2; that was resolved by
re-exporting the scaler under the pinned version and tuning the threshold from 0.15 to 0.70.

### C2 Beacon

After compromise, implants such as Cobalt Strike, Meterpreter, Sliver, and Brute Ratel
check in with their C2 at regular intervals. Even with jitter, this produces a periodicity
that human-driven traffic does not. The detector is dual-path. The temporal path is a
bidirectional LSTM over 100 consecutive flows between one source-destination pair, with four
features per timestep (inter-arrival time, packet size, byte count, direction). The spectral
path applies an FFT to the inter-arrival times and derives five features: FFT score,
dominant frequency, harmonic ratio, spectral entropy, and peak prominence.

Decision logic combines the two. A low-jitter beacon requires model probability above 90
percent with coefficient of variation below 0.05 and no match to a known benign periodic
pattern. An FFT-confirmed beacon requires probability above 90 percent with FFT score above
0.15, spectral entropy below 0.85, and peak prominence above 3.0. Known-benign periodic
traffic is explicitly excluded: NTP on port 123, TCP keepalives on 22/443/3389/5900, DNS
cache refresh on 53, and load-balancer health probes.

Status: the architecture and feature engineering are sound but the model has not yet been
validated against real botnet PCAP. CTU-13 Scenario 1 (Neris) is the recommended validation
target. The 100-flow activation threshold requires sustained capture, not brief snapshots.

### Encrypted Traffic

An FT-Transformer classifies encrypted flows into 14 application categories (chat, streaming,
file transfer, browsing, email, VoIP, P2P, and their VPN-tunneled variants) from 29 flow
metadata features, with no payload inspection. It reaches 88 percent accuracy. Its value is
encrypted-traffic visibility rather than a direct threat verdict: identifying traffic as VPN
is not itself malicious, but knowing what runs inside encrypted tunnels is operationally
useful in a SOC.

---

## Extraction pipeline

Four of the six models need features that CICFlowMeter cannot produce, so the pipeline is
hybrid and uses each tool where it is strongest.

| Requirement | CICFlowMeter | Custom Scapy extractor |
|---|---|---|
| Flow statistics for DDoS and Port Scan | Supported and required | Supported, but distribution differs |
| Individual DNS query strings for DGA | Not supported | Supported |
| 24 DNS lexical features for exfiltration | Not supported | Supported |
| 100-flow session time-series for C2 | Not supported | Supported |
| Real-time packet-by-packet live capture | Not supported | Supported |

**Phase 1, CICFlowMeter.** The PCAP is processed into flow features identical to those used
in training. A name-mapping wrapper translates the Python API output (`flow_byts_s`) to the
CIC CSV names the models expect (`Flow Bytes/s`). These events are tagged `cicflowmeter` and
routed only to DDoS and Port Scan.

**Phase 2, custom extractor.** The same PCAP is streamed packet by packet to produce DNS
events (to DGA and Exfiltration), flow events (to Encrypted Traffic), and session events, a
100-flow time-series per source-destination pair (to C2 Beacon).

### The covariate shift problem

When a model trained on CICFlowMeter features receives features computed by a different
implementation, the distributions diverge even when the feature names match. In testing, 92
percent of features showed statistically significant differences (Kolmogorov-Smirnov,
p < 0.05) between the custom extractor and CICFlowMeter, driven by differences in timeout
thresholds, TCP flag counting, and averaging windows. The resolution was to use CICFlowMeter
itself for the models trained on CICFlowMeter data, which removed the mismatch entirely.
This is why the routing rule described above is enforced strictly.

---

## Validated results

| Model | Evaluation dataset | Result | Method |
|---|---|---|---|
| DDoS | CIC-DDoS2019 test set, CIC-IDS-2017 PCAP | 99.97% F1, 0.998+ confidence on PCAP | XGBoost on CICFlowMeter features |
| DGA | DGArchive 2024, 137 families | 0.9787 macro-F1 | CNN-BiLSTM, char-level, family-wise split |
| Exfiltration | DNS-Tunnel-Datasets, 7 tools | 100% TPR, 3.3% FPR | VAE reconstruction error, threshold 0.70 |
| Port Scan | CIDDS-001 week2 validation + synthetic fast-scan PCAP | 82.6% detection, 7.4% FPR; 95.9% confidence on fast scans | RandomForest SPSD + fan-out backstop |
| C2 Beacon | Not yet validated | Pending | Requires sustained botnet PCAP (CTU-13) |
| Encrypted Traffic | Consolidated VPN dataset | 88% accuracy, 14 classes | FT-Transformer |

The numbers above are reported honestly, including where they are weak. The port-scan
82.6% figure is from a properly constrained RandomForest (an earlier unconstrained
DecisionTree scored higher but was completely broken — it predicted scan for all inputs
including benign traffic). The C2 model is presented as architecturally complete but not
yet empirically confirmed.

---

## Dashboard

The React frontend connects over WebSocket and renders threat data live.

- **Alert feed.** Chronological, severity-colored stream with source and destination IPs,
  threat type, confidence, and MITRE technique IDs. This is the primary triage surface.
- **3D threat graph.** Force-directed topology where nodes are IPs and edges are detected
  threat connections. When one internal host appears across DGA, C2, and exfiltration
  edges, the kill chain becomes visible. This is the payoff of running six models at once.
- **FFT spectrum.** Frequency-domain view of inter-arrival times for suspected C2 sessions.
  A sharp peak (for example 0.0167 Hz for a 60-second beacon) lets an analyst confirm
  periodicity independently of the model verdict.
- **Attack timeline.** When each threat type fired over the capture, revealing phases:
  reconnaissance early, then C2 establishment, then exfiltration.
- **MITRE heatmap.** Detected threats mapped onto ATT&CK tactics and techniques.
- **Traffic charts.** Bandwidth, packet rate, and protocol distribution for baseline context.

---

## How NetSentinel compares

**Versus Wireshark.** Wireshark inspects individual packets for an expert. NetSentinel is
automated detection. They are complementary: NetSentinel flags, Wireshark investigates.

**Versus Snort and Suricata.** Signature systems match known-bad patterns with low false
positives but need constant rule updates and miss zero-days and unseen C2 frameworks.
NetSentinel's behavioral models flag anomalous patterns regardless of whether the specific
tool has been seen before. A new tunneling tool still produces high-entropy DNS queries.

**Versus commercial platforms (Cisco Umbrella, CrowdStrike, Darktrace).** These are mature,
better-resourced products, and NetSentinel does not claim higher per-model accuracy. The
practical differences are deployment (it runs in air-gapped networks that cloud platforms
cannot), auditability (every model, feature, and boundary is inspectable), data sovereignty
(no traffic leaves the network), and cost (no licensing fees).

---

## Technology stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, uvicorn |
| Inference | ONNX Runtime, CPU execution |
| Packet processing | Scapy for dissection, CICFlowMeter for flow features |
| Frontend | React 19, Vite, Tailwind CSS |
| Visualization | Three.js for the 3D graph, Recharts for time-series and distributions |
| Real-time transport | WebSocket, server to client |
| Model distribution | HuggingFace Hub with local caching, offline capable |

---

## Installation

Prerequisites: Python 3.11, Node.js 18 or newer, and libpcap (Linux) or Npcap (Windows) for
live capture. PCAP-file analysis does not require capture drivers.

```
git clone https://github.com/qwertyuiopas17/wearecharliekirk.git
cd wearecharliekirk

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model weights are not stored in the repository. On first run the backend downloads them from
the HuggingFace repository `Unded-17/netsentinel-models` into `~/.cache/netsentinel/models`
and reuses that cache on every subsequent start. After the first fetch the system runs fully
offline. To pre-stage weights for an air-gapped host, copy that cache directory across, or
download the repository manually and place the files under the same path.

Frontend:

```
cd frontend
npm install
```

Note on scikit-learn: `requirements.txt` pins `scikit-learn==1.3.2`. This pin is load-bearing.
The exfiltration scaler must be loaded under the same version it was exported with; a mismatch
was the cause of the earlier 97.7 percent false-positive rate.

---

## Running the system

Backend, from the repository root:

```
python run.py
```

This serves the API and WebSocket on port 8000. On startup the model registry loads all six
models and the console prints the health, WebSocket, PCAP-upload, and live-capture endpoints.

Frontend, in a second terminal:

```
cd frontend
npm run dev
```

Open the URL Vite prints. The dashboard connects to `ws://localhost:8000/ws` automatically.

Analyze a capture by uploading a PCAP through the dashboard or by posting it to
`/api/pcap/upload`. For a live source, start capture through `/api/capture/start`. A built-in
traffic simulator can drive the dashboard when no capture is available.

---

## API reference

Base URL `http://localhost:8000`. Interactive OpenAPI docs are served at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness and model-load status |
| GET | `/api/alerts` | Recent alerts held in memory (most recent first) |
| GET | `/api/stats` | Aggregate detection statistics |
| POST | `/api/pcap/upload` | Upload a PCAP for offline analysis |
| POST | `/api/capture/start` | Start live capture on the configured interface |
| WS | `/ws` | Real-time alert and stats stream to the dashboard |

The WebSocket also accepts control messages from the client, for example
`{"action": "start_sim", "mode": "mixed"}` and `{"action": "stop_sim"}`, which drive the
traffic simulator.

---

## Configuration

All tunable constants live in `netsentinel/config.py`.

Detection thresholds (a model alerts when its confidence exceeds the threshold):

```
ddos               0.95     # the DDoS wrapper applies a stricter 98% gate plus a rate guard
c2_beacon          0.80
dga                0.70
encrypted_malware  0.70
port_scan          0.85     # conservative; the fan-out heuristic is the primary detector
exfiltration       0.70
```

Extraction and session parameters:

```
FLOW_IDLE_TIMEOUT     120    # seconds of inactivity before a flow is flushed
FLOW_ACTIVE_TIMEOUT   300    # maximum seconds a flow may stay open
SESSION_MIN_FLOWS     100    # flows per (src, dst) pair required before C2 detection runs
CAPTURE_INTERFACE     ...    # default capture interface; set per host and OS
```

Model paths, the HuggingFace repository ID, severity mapping, and the MITRE ATT&CK mapping
for all eight threat classes are also defined in this file.

---

## Project structure

```
wearecharliekirk/
├── run.py                          entry point, starts uvicorn on port 8000
├── requirements.txt                backend dependencies (scikit-learn pinned to 1.3.2)
├── netsentinel/
│   ├── main.py                     FastAPI app, startup model load, WebSocket, sim loop
│   ├── config.py                   paths, thresholds, severity/MITRE maps, PortScanSettings
│   ├── models/
│   │   ├── registry.py             loads and holds all six ONNX models
│   │   ├── ddos.py                 XGBoost DDoS detector
│   │   ├── port_scan.py            per-flow XGBoost (legacy, supplementary only)
│   │   ├── portscan_detector.py    SPSD/UPSD network-event classifier + fan-out backstop
│   │   ├── dga.py                  CNN-BiLSTM domain classifier
│   │   ├── exfiltration.py         VAE reconstruction-error detector
│   │   ├── c2_beacon.py            BiLSTM plus FFT beacon detector
│   │   ├── encrypted.py            FT-Transformer traffic classifier
│   │   └── weights/
│   │       └── portscan_spsd_decisiontree.pkl  trained RandomForest (20 trees)
│   ├── extractor/
│   │   ├── cicflowmeter_wrapper.py Phase 1 flow-feature extraction
│   │   ├── network_event_builder.py SPSD feature aggregation (6 behavioral features)
│   │   ├── flow_extractor.py       custom flow summaries
│   │   ├── dns_extractor.py        DNS query parsing
│   │   ├── dns_feature_builder.py  24 DNS lexical features
│   │   ├── session_builder.py      100-flow session time-series
│   │   ├── pcap_reader.py          PCAP ingestion
│   │   └── unsw_feature_builder.py legacy feature mapping
│   ├── netinfo/
│   │   ├── network_info.py         network topology loader + lookups
│   │   ├── network_info.json       production network topology
│   │   ├── network_info_cidds.json CIDDS-001 topology (training)
│   │   └── network_info_cicids2017.json  CICIDS2017 topology (testing)
│   ├── pipeline/
│   │   ├── analyzer.py             routing, correlation, port scan batch processing
│   │   ├── portscan_integration.py PortScanRouter batch interface
│   │   └── alert_manager.py        severity assignment and alert retention
│   ├── integrity/                  forensic chain-of-custody (Phase 1 complete)
│   │   ├── encoding.py             JCS canonicalization, PPM scores
│   │   ├── envelope.py             DSSE envelope construction
│   │   ├── receipt.py              in-toto Statement v1 receipts
│   │   ├── merkle.py               domain-separated Merkle tree + consistency proofs
│   │   ├── ledger.py               Ed25519 append-only transparency log
│   │   ├── anchor_service.py       windowed batching + Git anchor backend
│   │   ├── claims.py               9-claim verifier (7 live, 2 stubbed)
│   │   ├── replay.py               deterministic inference replay
│   │   ├── blobstore.py            evidence blob storage
│   │   ├── releases.py             signed model release registry
│   │   └── codehash.py             pipeline code digest
│   ├── api/
│   │   ├── routes.py               REST + integrity endpoints
│   │   └── websocket.py            WebSocket hub
│   └── simulator/
│       └── traffic_gen.py          synthetic traffic for demos
├── scripts/
│   ├── train_spsd.py               CIDDS-001 SPSD model training
│   └── retrain_combined.py         combined CIDDS + synthetic retraining
├── frontend/                       React 19 + Vite dashboard
├── tests/
│   └── test_portscan.py            port scan detection test suite (5 tests)
├── tools/
│   └── netsentinel_verify.py       CLI integrity verifier
└── dgatrain.ipynb                  DGA model training notebook
```

---

## 🔒 Forensic Integrity Layer — Proof-Carrying Alerts

NetSentinel includes an **optional tamper-evident integrity layer** for forensic
chain-of-custody, designed for high-assurance environments where cryptographic verification
of detection provenance is required.

**Key point:** This is **not** "blockchain for detection" (a misuse of the technology). It
is a **Certificate Transparency-style append-only log** for forensic integrity — the one
defensible use of blockchain patterns in intrusion detection. Detection is ML/heuristics;
blockchain adds nothing to detection. The value is making any post-hoc tampering of alerts,
models, or evidence **cryptographically detectable and provable**.

### Quick Start

Enable with environment variable:
```bash
export INTEGRITY_ENABLED=true
python -m netsentinel.main
```

Visit the dashboard → click any alert → **"Verify" tab** → see cryptographic verification
with expandable claim details.

### Architecture

Every alert receives a **DSSE-signed receipt** (in-toto Statement v1) containing
cryptographic commitments to:
- Evidence digest (SHA-256 of alert payload)
- Feature vector digest (the exact inputs the model saw)
- Model digest (ONNX file hash)
- Policy digest (thresholds + decision gates)
- Decision (class, score in parts-per-million, threshold)

Receipts are Merkle-batched into 60-second windows and anchored to an append-only
transparency log with Ed25519-signed Signed Tree Heads. The 9-claim verifier produces
PASS/FAIL/UNVERIFIABLE verdicts for each guarantee. Any tampering — alert edits, model
swaps, threshold changes — is cryptographically detectable.

### Capabilities

**1. DSSE-Signed Receipts** ✅
Every alert is wrapped in a [DSSE](https://github.com/secure-systems-lab/dsse) envelope
(in-toto Statement v1) signed with Ed25519. The receipt cryptographically binds the alert
payload, the feature vector the model saw, the model file hash, and the decision
thresholds into a single tamper-evident object. If anyone edits the alert after the fact —
changes a confidence score, swaps a source IP, or deletes evidence — the signature
verification fails and the tampering is provable.

**2. Deterministic Replay** ✅
The exact feature vector and model binary are stored alongside each receipt. An independent
verifier can re-run inference on the stored inputs and confirm the model produces the same
verdict. This proves the alert was not fabricated — the model genuinely made that decision
on that data. XGBoost models (DDoS, port scan) replay bit-exact; neural models (DGA, C2,
encrypted, VAE) are pinned to single-threaded deterministic ONNX Runtime.

**3. Merkle-Batched Transparency Log** ✅
Receipts are batched into 60-second windows and hashed into a Merkle tree. The tree root
is signed as a Signed Tree Head (STH) and appended to an Ed25519-signed, hash-chained
ledger. This is the same architecture as Certificate Transparency (used at internet scale
by browser vendors). A log of 80 million events requires only 3 KB of proof to verify
append-only-ness (Crosby & Wallach). Any attempt to silently delete, reorder, or insert
alerts breaks the Merkle inclusion proof or the hash chain.

**4. Signed Model Release Registry** ✅
Every ONNX model loaded at startup must appear in a signed release log (`releases.jsonl`).
Each entry records the model digest, version, feature schema digest, and a separate
release-key Ed25519 signature (not the sensor key — so a compromised sensor cannot forge
model approvals). If an attacker swaps the ONNX file on disk, the digest mismatch is
detected at load time and the verification claim flips to FAIL. Supports revocation and
rollback detection.

**5. Pipeline Code Digest** ✅
A deterministic hash of the code files that could change predictions (extractors, model
wrappers, encoding module) is committed at startup and checked on every verification.
If anyone modifies an extractor or wrapper — even adding a single line — the code digest
changes and the pipeline-attestation claim fails. Includes git commit reference and
dirty-flag detection.

**6. Frozen Feature Schemas** ✅
Each model's expected feature names are frozen at first run into JSON schema files. On
every prediction, the live feature vector is checked against the frozen schema. Feature
drift (a renamed column, a missing field, a new feature injected) is cryptographically
detectable — closing the exact class of bug (covariate shift from mismatched features)
that broke NetSentinel's original port scan model.

**7. External Timestamping (Git Anchor)** ✅
Periodic checkpoints of the ledger root are committed to a Git repository, providing an
independent timestamp anchor that cannot be backdated without rewriting Git history. This
works fully offline (air-gapped networks). The checkpoint proves that a specific set of
alerts existed at a specific time.

**8. 9-Claim Verifier + Dashboard Panel** ✅
A structured verifier evaluates nine cryptographic guarantees per alert and returns
PASS/FAIL/UNVERIFIABLE for each. The React dashboard surfaces this as an interactive
verification panel — click any alert → "Verify" tab → see each claim with expandable
details. A CLI verifier (`tools/netsentinel_verify.py`) provides the same checks from the
command line for forensic workflows.

**9. Evidence Blob Store** ✅
Raw feature vectors and model inputs are persisted in a content-addressed blob store,
keyed by SHA-256 digest. This ensures the exact bytes the model saw are preserved for
replay and audit, even if the live data stream has moved on. Configurable retention
(default: 30 days).

**10. Witness Protocol (Multi-Party Fork Detection)** 🔜
An independent witness server cosigns ledger checkpoints. Even if an attacker compromises
the sensor's signing key and rewrites the entire ledger, the witness holds a cached copy
of the last checkpoint it cosigned. When the rewritten ledger is presented, the witness
detects the fork (divergent tree roots for the same sequence number) and raises a
`FORK_DETECTED` alert with cryptographic proof. This is the strongest guarantee in the
system: it defeats the "compromised host" threat model that no single-machine integrity
layer can handle alone.

**11. Completeness Monitor** 🔜
Monitors the ledger for gaps (missing sequence numbers), forks (divergent hash chains),
and stalls (no new blocks beyond 2× the expected window). Raises `COMPLETENESS_ALERT`
events over WebSocket. This catches the **omission attack** — an adversary who deletes
alerts rather than editing them. Detecting that alerts are *missing* is fundamentally
harder than detecting that they were *changed*, and this component addresses it.

**12. RFC 3161 Timestamping** 🔜
High-strength external timestamps from a PKI-signed Time Stamping Authority (TSA). One
HTTPS round-trip returns an independently verifiable, legally recognized timestamp token.
Unlike Git anchors (which depend on the repository owner not rewriting history), RFC 3161
tokens are signed by an independent CA and hold up under standards like eIDAS and FRE
901/902. Works when network connectivity is available; queues and retries when offline.

**13. SQLite Sealing** 🔜
Crash-safe monotonic receipt counter using a single-row SQLite table with transactional
increment. Prevents receipt sequence reuse after an unclean shutdown — the one failure
mode that the current file-based counter cannot fully guard against. Replaces
`seq_state.json` with atomic guarantees.

**14. Selective Disclosure** 🔜
Per-field salted commitments allow sharing specific evidence fields with a third party
(e.g., an incident response team or a court) without revealing the entire alert. The
recipient receives a Merkle proof that the disclosed fields are genuine subsets of the
committed receipt — privacy-preserving evidence sharing.

**15. Analyst Action Chain** 🔜
Every human decision on an alert (acknowledge, escalate, close, add notes) is signed and
appended to a per-alert action chain. This extends the chain of custody from automated
detection through human triage — proving not just that the system detected a threat, but
who saw it, when they acted, and what they decided.

**16. Zero-Knowledge ML Proofs (ZKML)** 🔜
Cryptographic proof that a specific model produced a specific output on specific inputs,
without revealing the model weights or the full input data. This is the theoretical
end-state for model provenance — proving correctness without exposing intellectual
property. Research-stage; depends on ZKML toolchain maturity.

**17. HSM/KMS Key Management** 🔜
Hardware Security Module or cloud KMS integration for signing key storage. File-based
Ed25519 keys are adequate for demonstration but not for production deployments where the
host may be compromised. HSM-backed keys ensure the signing key cannot be extracted even
with root access.

### The Nine Verification Claims

| # | Claim | What it proves | Status |
|---|-------|----------------|--------|
| 1 | Evidence integrity | Alert payload not edited after issuance | ✅ |
| 2 | Feature integrity | Model saw exactly these input features | ✅ |
| 3 | Approved model | ONNX file matches signed release registry | ✅ |
| 4 | Approved pipeline | Code digest matches committed baseline | ✅ |
| 5 | Deterministic replay | Re-running inference reproduces the same verdict | ✅ |
| 6 | Policy verification | Thresholds and decision gates locked to receipt | ✅ |
| 7 | Merkle inclusion | Receipt is provably included in a batched window | ✅ |
| 8 | Ledger consistency | Hash-chained blocks, Ed25519 signed, append-only | ✅ |
| 9 | External anchoring | Checkpoint anchored to independent timestamp | ✅ (Git) |

### Industry Precedent

The architecture follows established patterns used at scale, not hype:

| System | What it does | Relevance |
|--------|-------------|-----------|
| **Guardtime KSI** | National-scale tamper-evident logging for Estonia's e-governance and health records | Same Merkle-batch + anchor pattern; "truth over trust" |
| **Certificate Transparency** | Largest production append-only Merkle log, run by browser vendors | Direct architectural model for our transparency log |
| **LogStamping (2025)** | Hybrid on-chain/off-chain audit trails — full logs off-chain, only hashes on ledger | Validates our "hash-only on ledger" design |
| **B-CoC** | Blockchain-based chain of custody for digital forensics | Academic precedent for exactly our use case |
| **OpenTimestamps** | Free, standardized Bitcoin-anchored Merkle root timestamping | Roadmapped for maximum-independence anchoring |

### What We Deliberately Did Not Build

| Anti-pattern | Why it is wrong |
|---|---|
| Detect attacks using blockchain | Detection is ML/heuristics. Blockchain adds nothing to detection quality. |
| Store packets/flows on-chain | Infeasible at network speeds. Hyperledger Fabric caps ~70 TPS for write-heavy workloads. |
| Consensus-based detection across nodes | Massive scope, no benefit for a single-sensor deployment. |

Stating what we deliberately scoped out — and citing the literature that documents these
anti-patterns — demonstrates deeper understanding than claiming a broad blockchain-IDS
integration.

**Design documents:** `docs/middle.md` (base layer threat model and research),
`PROOF_CARRYING_ALERTS_IMPLEMENTATION_PLAN.md` (provenance upgrade), `docs/kj.md`
(full integration plan).

**Legal positioning:**
> NetSentinel produces technical evidence that may support authentication, integrity
> analysis, and chain-of-custody testimony. Cryptographic anchoring demonstrates that
> specific bytes existed at a specific time and were not subsequently altered; it does not
> by itself establish that an alert's underlying claim is true, nor does it guarantee legal
> admissibility, which remains a determination for the court.


---

## Known limitations

The system is honest about its boundaries.

- **Detection only.** NetSentinel does not block, quarantine, or otherwise act on traffic.
  It is a sensor, not an enforcement point.
- **Port-scan detection rate is 82.6%, not 99%.** The earlier 99% figure came from a broken
  overfit DecisionTree that predicted scan for all inputs including benign traffic. The
  retrained RandomForest correctly distinguishes benign from malicious at the cost of lower
  recall. Ultra-slow scans (<5 ports/minute) are caught by the fan-out backstop, not the
  ML model.
- **Port-scan topology dependency.** The NEIP and NETCP features require accurate
  `network_info.json`. Without it, the system degrades to 4 of 6 features plus fan-out.
- **C2 beacon is not yet empirically validated.** The architecture is complete but needs
  validation against sustained real botnet PCAP such as CTU-13 Scenario 1.
- **C2 requires sustained capture.** The 100-flow activation threshold means brief snapshots
  will not trigger beacon detection.
- **Encrypted-traffic classification is not a threat verdict.** VPN usage is legitimate; the
  model provides visibility, not an accusation.
- **Scaler version coupling.** The exfiltration scaler is tied to scikit-learn 1.3.2. Do not
  bump that pin without re-exporting the scaler.
- **Integrity layer keys are file-based.** Ed25519 signing keys are stored on disk with
  filesystem permissions. HSM/KMS integration is roadmapped for Phase 3.

---

## Roadmap

With the pipeline now validated (covariate shift eliminated, exfiltration at 100 percent TPR
and 3.3 percent FPR, DGA corrected to an honest 0.9787 macro-F1, port scan retrained and
functional), the foundation is stable enough to extend.

**Detection priorities:**
- Validate C2 beacon detection against CTU-13 and other labeled botnet captures.
- Add a Legitimate Service Abuse detector for C2 and exfiltration that ride trusted services
  (Telegram bots, cloud storage, chat webhooks), where domain blocking is not viable.
- Implement port-scan succession decay (instead of reset) for better ultra-sparse scan
  tracking without relying on the fan-out backstop.
- Broaden live-capture coverage and interface auto-detection across Linux and Windows.

**Integrity priorities (Phase 2–3):**
- SQLite sealing for crash-safe receipt counters.
- Completeness monitor for gap, fork, and stall detection.
- Witness protocol for multi-party ledger cosigning and fork detection.
- RFC 3161 external timestamping for high-strength anchoring when connected.
- Selective disclosure and analyst action chains (Phase 3).
- ZKML and HSM/KMS integration (Phase 3).

---

## License

See [LICENSE](LICENSE).
