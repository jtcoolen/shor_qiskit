"""physical.py: Gidney 2025's surface-code model, reproduced from its inputs."""
import physical as PH


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def main():
    code = PH.SurfaceCode()
    assert code.distance() == 25
    assert code.hot_per_logical() == 1352
    ok("d = 25 is the smallest distance reaching 1e-15 at p = 1e-3; 1352 qubits "
       "per hot patch")

    cult, surg, tot = code.ccz_rounds()
    assert abs(cult - 14.8) < 0.1 and abs(surg - 100) < 1e-9 and abs(tot - 114.8) < 0.2
    assert code.ccz_period_us() == 25.0
    ok(f"CCZ: {cult:.1f} cultivation + {surg:.0f} surgery = {tot:.1f} rounds "
       f"(paper: 14.7 + 100 = 114.7), 150 with slack -> one CCZ per 25 us")

    r = PH.estimate(PH.g25_rsa2048(), code)
    assert r["physical_total"] == 897864, r["physical_total"]
    assert r["physical"] == {"cold": 550400, "hot": 177112, "compute": 170352}
    assert r["logical_including_idle"] == 1537
    assert abs(r["no_error_rate"] - 0.933) < 0.002, r["no_error_rate"]
    assert abs(r["days"] - 4.96) < 0.01, r["days"]
    ok(f"RSA-2048: {r['physical_total']:,} physical qubits (1280 x 430 + 131 x "
       f"1352 + 126 x 1352), no-error rate {r['no_error_rate']:.3f}, "
       f"{r['days']:.2f} days -- Gidney 2025's figures exactly")
    rf = PH.estimate(PH.g25_rsa2048(hot="formula"), code)
    assert rf["physical_total"] == 926256
    ok("with the paper's own hot-qubit formula (3f + 2l + len m = 152, not the "
       "131 it states): 926,256 -- the inconsistency is in the source")

    # a measured ECDLP run through the same model: CCZ-throughput limited
    e = PH.estimate(PH.ecdlp_profile("ECDLP-256", 1500, 4.7e7))
    minutes = e["hours_per_shot"] * 60
    assert e["runtime_model"] == "CCZ-throughput" and 15 < minutes < 25
    ok(f"ECDLP-256 at 47M Toffolis / 1500 logical qubits: {minutes:.0f} minutes "
       f"per run, CCZ-throughput limited (Babbush et al.: 18-23 minutes); "
       f"{e['physical_total']:,.0f} physical qubits with every logical qubit in "
       f"hot patches -- Babbush et al.'s <500k assumes denser storage")


def test_physical():
    main()


if __name__ == "__main__":
    main()
    print("\ntest_physical: all passed")
