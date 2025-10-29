# scripts/skywire_client.py
from __future__ import annotations
import os
import requests
from typing import Dict, Any


class SkywireClient:
    def __init__(self, mode: str = "http", http_base: str = "https://sd.skycoin.com", timeout_s: int = 8):
        self.mode = mode
        self.http_base = http_base.rstrip("/")
        self.timeout_s = timeout_s

    def ping_public(self) -> Dict[str, Any]:
        """
        Ping minimal: on requête des endpoints publics pour dériver des features de base.
        Retourne { proxies, vpn, transports, dmsg_entries, rf_status_ok }
        """
        if self.mode == "mock":
            return {
                "proxies": 1000,
                "vpn": 500,
                "transports": 1500,
                "dmsg_entries": 2500,
                "rf_status_ok": 0,  # rf 404 => 0
            }

        try:
            r1 = requests.get(f"{self.http_base}/api/services?type=proxy", timeout=self.timeout_s)
            r2 = requests.get(f"{self.http_base}/api/services?type=vpn", timeout=self.timeout_s)
            r3 = requests.get("https://tpd.skywire.skycoin.com/all-transports", timeout=self.timeout_s)
            r4 = requests.get("https://dmsgd.skywire.skycoin.com/dmsg-discovery/entries", timeout=self.timeout_s)
            rf = requests.get("https://rf.skywire.skycoin.com/", timeout=self.timeout_s)
            proxies = len(r1.json()) if r1.ok else 0
            vpn = len(r2.json()) if r2.ok else 0
            transports = len(r3.json()) if r3.ok else 0
            dmsg_entries = len(r4.json()) if r4.ok else 0
            rf_ok = 1 if rf.status_code == 200 else 0
            return {
                "proxies": proxies,
                "vpn": vpn,
                "transports": transports,
                "dmsg_entries": dmsg_entries,
                "rf_status_ok": rf_ok,
            }
        except Exception:
            # en cas d’échec réseau => valeurs neutres
            return {"proxies": 0, "vpn": 0, "transports": 0, "dmsg_entries": 0, "rf_status_ok": 0}
