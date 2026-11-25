# op 层直测: 注册拦截 / 四模式命中 / 与原生对照 / autograd / gradcheck /
# 概率图消费者 / 错误路径 / F.sdpa(MATH) 反向消费（返回 metrics）。
# 运行: python3 test/op_level.py [ascend910|cpu]
from __future__ import annotations

import sys
from pathlib import Path

import torch

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def _mk(B, Hq, Hkv, S, D, dev, seed=0, dtype=torch.float32, mask=None,
        req_grad=True):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, Hq, S, D, generator=g, dtype=dtype).to(
        dev).requires_grad_(req_grad)
    k = torch.randn(B, Hkv, S, D, generator=g, dtype=dtype).to(
        dev).requires_grad_(req_grad)
    v = torch.randn(B, Hkv, S, D, generator=g, dtype=dtype).to(
        dev).requires_grad_(req_grad)
    m = None
    if mask == "bool":
        m = (torch.rand(B, Hq, S, S, generator=g) > 0.3).to(dev)
    elif mask == "float":
        m = (torch.randn(B, Hq, S, S, generator=g) * 0.1).to(
            dev, dtype).requires_grad_(req_grad)
    return q, k, v, m


def _op(q, k, v, m=None, p=0.0, causal=False, dm=None, scale=None, gqa=False):
    return torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, m, p, causal, dm, scale=scale, enable_gqa=gqa)


def _cpu_qkv(B, Hq, Hkv, S, D, mask=None, seed=7):
    import torch
    g = torch.Generator("cpu").manual_seed(seed)
    q = torch.randn(B, Hq, S, D, generator=g, dtype=torch.float64,
                    requires_grad=True)
    k = torch.randn(B, Hkv, S, D, generator=g, dtype=torch.float64,
                    requires_grad=True)
    v = torch.randn(B, Hkv, S, D, generator=g, dtype=torch.float64,
                    requires_grad=True)
    mm = None
    if mask == "float":
        mm = (torch.randn(B, Hq, S, S, generator=g, dtype=torch.float64)
              * 0.1).requires_grad_(True)
    return q, k, v, mm


def run(profile):
    import torch
    import torch.nn.functional as F
    from torch.autograd import gradcheck
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from kernel.triton_level import sdpa_math_triton
    from kernel.torch_level import sdpa_math_torch
    from kernel.p800_fast_level import sdpa_math_p800_fast
    from kernel.thead_fast_level import sdpa_math_thead_fast
    from kernel.mlu590_fast_level import sdpa_math_mlu590_fast
    from reference import make_dropout_mask, sdpa_math_reference
    from register import register_a1

    dev = profile.torch_device
    results = []

    def check(name, cond, extra=""):
        mark = "OK  " if cond else "FAIL"
        results.append(f"  [{mark}] {name}" + (f" — {extra}" if extra else ""))
        assert cond, f"{name}: {extra}"

    # ── 0) 注册前抓原生基线（注册后 torch.ops 即本实现，抓不到原生）──
    # 掩码用 float: native **直调** 时 bool mask 按 0/1 加性怪癖（已知唯一
    # 有意分歧，见 reference 顶注），不能当基线；float 加性两边一致。
    q, k, v, m = _mk(2, 4, 4, 64, 64, dev, seed=1, mask="float",
                     req_grad=False)
    base_out, base_p = _op(q, k, v, m, 0.0, False, None)
    base_out, base_p = base_out.clone(), base_p.clone()

    # P800 fused backward probe: native gradient is captured before the A1
    # registration takes over this op.  This no-mask causal GQA case is the
    # production fast path (unlike the float-mask case above).
    fast_baseline = None
    if profile.vendor == "kunlunxin":
        qf, kf, vf, _ = _mk(1, 8, 2, 96, 64, dev, seed=21,
                            dtype=torch.float16)
        f_out, f_p = _op(qf, kf, vf, None, 0.0, True, None, gqa=True)
        f_do, f_dp = torch.randn_like(f_out), torch.randn_like(f_p)
        fast_baseline = torch.autograd.grad(
            (f_out, f_p), (qf, kf, vf), (f_do, f_dp),
            retain_graph=False,
        )
        fast_baseline = tuple(x.detach().clone() for x in fast_baseline)
        fast_qkv = tuple(x.detach().clone().requires_grad_(True)
                         for x in (qf, kf, vf))

    # ── 1) 注册: profile 键 + CPU 键（gradcheck / F.sdpa 走 torch impl）──
    # 注册返回的 lib 必须持有引用（否则注册被 GC 掉，见 register.py）
    ctr_main: dict = {"n": 0}
    libs_main = register_a1(profile.dispatch_key, ctr_main,
                            profile.default_impl)
    ctr_cpu: dict = {"n": 0}
    libs_cpu = (register_a1("CPU", ctr_cpu, "torch")
                if profile.dispatch_key != "CPU" else None)

    # ── 2) 拦截计数 × 四模式 ──
    q2, k2, v2, _ = _mk(1, 4, 4, 32, 32, dev, seed=2, req_grad=False)
    n0 = ctr_main["n"]
    out1, p1 = _op(q, k, v, m, 0.0, False, None)
    check("grad 开启: torch.ops 直调被拦截", ctr_main["n"] > n0,
          f"+{ctr_main['n'] - n0}")

    n0 = ctr_main["n"]
    with torch.no_grad():
        _op(q2, k2, v2, None, 0.0, True, None)
    check("no_grad: 被拦截（impl_fn_ 直通）", ctr_main["n"] > n0,
          f"+{ctr_main['n'] - n0}")

    n0 = ctr_main["n"]
    with torch.inference_mode():
        out_inf, _ = _op(q2, k2, v2, None, 0.0, True, None)
    check("inference_mode: 被拦截（Autograd*/plain 成对注册回归）",
          ctr_main["n"] > n0, f"+{ctr_main['n'] - n0}")
    check("inference_mode 输出有限", bool(torch.isfinite(out_inf).all()))

    n0 = ctr_main["n"]
    _op(q2 + 0.1, k2, v2, None, 0.0, True, None)
    check("无 requires_grad 张量: 被拦截", ctr_main["n"] > n0,
          f"+{ctr_main['n'] - n0}")

    # ── 3) 与原生基线对照 + A1 包装与直接 impl 逐位一致 ──
    eo = (out1 - base_out).abs().max().item()
    ep = (p1 - base_p).abs().max().item()
    check("拦截后输出 = 原生基线", max(eo, ep) < 1e-5,
          f"err={max(eo, ep):.2e}")

    impl_fn = {
        "triton": sdpa_math_triton,
        "p800": sdpa_math_p800_fast,
        "thead": sdpa_math_thead_fast,
        "mlu590": sdpa_math_mlu590_fast, 
    }.get(profile.default_impl, sdpa_math_torch)
    d_out, d_p = impl_fn(q, k, v, m, 0.0, False, None)
    check("A1 包装 = 直接调 impl（逐位）",
          torch.equal(out1, d_out) and torch.equal(p1, d_p))

    # ── 3b) MLU590 fast path: fp16/bf16 显式覆盖（评审 P2-1）──
    # fast path 的 _fast_eligible 只接 fp16/bf16；默认 _mk 是 fp32，
    # 不显式覆盖的话会全部 fallback 到 sdpa_math_torch，证明不了 fast
    # path 正确。direct requires_grad 按设计回退 torch，故 req_grad=False。
    if profile.default_impl == "mlu590":
        from kernel.mlu590_fast_level import _fast_eligible
        for dt in (torch.float16, torch.bfloat16):
            dt_name = str(dt).split(".")[-1]
            qf, kf, vf, _ = _mk(1, 4, 4, 64, 64, dev, seed=31,
                                dtype=dt, req_grad=False)
            elig = _fast_eligible(qf, kf, vf, None, 0.0, False, None, None)
            check(f"MLU fast eligible: {dt_name}", elig)
            if not elig:
                continue

            d_out, d_p = sdpa_math_mlu590_fast(qf, kf, vf)
            r_out, r_p = sdpa_math_torch(qf, kf, vf)
            eo_f = (d_out.float() - r_out.float()).abs().max().item()
            ep_f = (d_p.float() - r_p.float()).abs().max().item()
            rs = d_p.float().sum(-1)
            check(f"MLU fast direct out/P: {dt_name}",
                  max(eo_f, ep_f) < 2e-2,
                  f"out={eo_f:.2e} P={ep_f:.2e}")
            # bf16 只有 8 位尾数（eps≈3.9e-3），P 行和天然有 ~1e-3 量级
            # 的量化偏差；这是数值精度，不是 P1-1 的回归（P1-1 的偏差是
            # 25% 量级）。fp16 保持 1e-3 严格阈值，bf16 放宽到 5e-3。
            row_sum_tol = 1e-3 if dt == torch.float16 else 5e-3
            check(f"MLU fast P row_sum: {dt_name}",
                  bool(((rs - 1).abs() < row_sum_tol).all()),
                  f"min={rs.min():.4f} max={rs.max():.4f} "
                  f"tol={row_sum_tol:.0e}")

            n0 = ctr_main["n"]
            with torch.no_grad():
                _op(qf, kf, vf, None, 0.0, False, None)
            check(f"MLU fast no_grad 拦截: {dt_name}",
                  ctr_main["n"] > n0, f"+{ctr_main['n'] - n0}")

            n0 = ctr_main["n"]
            with torch.inference_mode():
                _op(qf, kf, vf, None, 0.0, False, None)
            check(f"MLU fast inference_mode 拦截: {dt_name}",
                  ctr_main["n"] > n0, f"+{ctr_main['n'] - n0}")

        # ── 3c) dropout 消费 fast path 产出的 P ──
        # A1 的 forward 以 dropout_p=0 调 fast path 拿 P，Function 层施加
        # dropout 再用 P 重算 O。这里构造 dropout_p>0 的 A1 调用，显式传
        # dropout_mask 保证确定性，对比 torch reference。
        for dt in (torch.float16, torch.bfloat16):
            dt_name = str(dt).split(".")[-1]
            qd, kd, vd, _ = _mk(1, 4, 4, 64, 64, dev, seed=41,
                                dtype=dt, req_grad=False)
            dm = make_dropout_mask((1, 4, 64, 64), 0.3, dev)
            d_out, d_p = _op(qd, kd, vd, None, 0.3, False, dm)
            r_out, r_p = sdpa_math_torch(
                qd, kd, vd, None, 0.3, False, dm)
            eo_d = (d_out.float() - r_out.float()).abs().max().item()
            ep_d = (d_p.float() - r_p.float()).abs().max().item()
            check(f"MLU fast A1+dropout 消费 P: {dt_name}",
                  max(eo_d, ep_d) < 2e-2,
                  f"out={eo_d:.2e} P={ep_d:.2e}")

        # ── 3d) overflow 场景两路梯度（走 A1）──
        # direct requires_grad 按设计 fallback torch，故走 A1：
        # forward 在 no_grad 下用 fast path 拿 O/P，backward 数学实现。
        # 检查 dO/dP 两路梯度有限且与 reference 一致。
        for dt in (torch.float16, torch.bfloat16):
            dt_name = str(dt).split(".")[-1]
            D, scale = 64, 1.0 / 8.0
            qg = torch.full((1, 4, 2, D), 40.0, dtype=dt, device=dev,
                            requires_grad=True)
            kg = torch.full((1, 4, 2, D), 40.0, dtype=dt, device=dev,
                            requires_grad=True)
            vg = torch.randn(1, 4, 2, D, dtype=dt, device=dev,
                             requires_grad=True)
            o_g, p_g = _op(qg, kg, vg, None, 0.0, False, None, scale=scale)
            do = torch.randn_like(o_g)
            dp = torch.randn_like(p_g)
            go = torch.autograd.grad(o_g, (qg, kg, vg), do, retain_graph=True)
            gp = torch.autograd.grad(p_g, (qg, kg, vg), dp)
            check(f"MLU fast A1 overflow dO 有限: {dt_name}",
                  all(bool(torch.isfinite(t).all()) for t in go))
            check(f"MLU fast A1 overflow dP 有限: {dt_name}",
                  all(bool(torch.isfinite(t).all()) for t in gp))

            # reference 同 A1 模式（CPU 键 + torch impl）
            qr = qg.detach().clone().requires_grad_(True)
            kr = kg.detach().clone().requires_grad_(True)
            vr = vg.detach().clone().requires_grad_(True)
            r_out, r_p = sdpa_math_torch(qr, kr, vr, None, 0.0, False, None,
                                          scale=scale)
            gro = torch.autograd.grad(r_out, (qr, kr, vr), do,
                                      retain_graph=True)
            grp = torch.autograd.grad(r_p, (qr, kr, vr), dp,
                                      allow_unused=True)
            grp = tuple(torch.zeros_like(t) if x is None else x
                        for x, t in zip(grp, (qr, kr, vr)))
            e_go = max((a.float() - b.float()).abs().max().item()
                       for a, b in zip(go, gro))
            e_gp = max((a.float() - b.float()).abs().max().item()
                       for a, b in zip(gp, grp))
            check(f"MLU fast A1 overflow dO = reference: {dt_name}",
                  e_go < 5e-2, f"max={e_go:.2e}")
            check(f"MLU fast A1 overflow dP = reference: {dt_name}",
                  e_gp < 5e-2, f"max={e_gp:.2e}")

    # ── 4) autograd: grad_fn + backward + 梯度 = 参考图反传 ──
    qa, ka, va, ma = _mk(1, 4, 4, 48, 48, dev, seed=3, mask="float")
    # float mask 与 causal 互斥（native 冲突规则，不论 mask 类型）
    out, probs = _op(qa, ka, va, ma, 0.0, False, None)
    check("requires_grad → 有 grad_fn",
          out.requires_grad and out.grad_fn is not None)
    g = torch.randn_like(out)
    gprobs = torch.randn_like(probs)
    go = torch.autograd.grad(out, (qa, ka, va, ma), g, retain_graph=True)
    gp = torch.autograd.grad(probs, (qa, ka, va, ma), gprobs)
    check("backward: out/P 两路梯度均有限",
          all(bool(torch.isfinite(t).all()) for t in go + gp))

    # 同种子参考图反传对照
    qf, kf, vf, mf = _mk(1, 4, 4, 48, 48, dev, seed=3, mask="float")
    r_out, r_p = sdpa_math_reference(qf, kf, vf, mf, 0.0, False, None)
    ref_go = torch.autograd.grad(r_out, (qf, kf, vf, mf), g, retain_graph=True)
    # P 图本身不依赖 V → 参考图对 V 的反传是 unused（allow_unused），
    # 对齐到 0 后再比（本算子 backward 对 dout=None 返回零 dV，等价）
    ref_gp = torch.autograd.grad(r_p, (qf, kf, vf, mf), gprobs,
                                 allow_unused=True)
    ref_gp = tuple(torch.zeros_like(t) if x is None else x
                   for x, t in zip(ref_gp, (qf, kf, vf, mf)))
    e_go = max((a - b.detach()).abs().max().item() for a, b in zip(go, ref_go))
    e_gp = max((a - b.detach()).abs().max().item() for a, b in zip(gp, ref_gp))
    check("out-消费者梯度 = 参考反传", e_go < 1e-4, f"max={e_go:.2e}")
    check("P-消费者梯度 = 参考反传", e_gp < 1e-4, f"max={e_gp:.2e}")

    if fast_baseline is not None:
        qf, kf, vf = fast_qkv
        f_out, f_p = _op(qf, kf, vf, None, 0.0, True, None, gqa=True)
        got_fast = torch.autograd.grad(
            (f_out, f_p), (qf, kf, vf), (f_do, f_dp), retain_graph=False,
        )
        fast_err = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in zip(got_fast, fast_baseline)
        )
        check("P800 fused backward = native", fast_err < 0.01,
              f"max={fast_err:.2e}")

    # ── 5) gradcheck (fp64, CPU 键 → torch impl) ──
    def gc(name, make, fn, with_mask_grad=False):
        qg, kg, vg, mg = make()
        try:
            if with_mask_grad:
                gradcheck(lambda a, b, c, mm: fn(a, b, c, mm),
                          (qg, kg, vg, mg), eps=1e-6, atol=1e-5, rtol=1e-3)
            else:
                gradcheck(lambda a, b, c: fn(a, b, c, mg),
                          (qg, kg, vg), eps=1e-6, atol=1e-5, rtol=1e-3)
            check(f"gradcheck: {name}", True)
        except Exception as e:  # noqa: BLE001
            check(f"gradcheck: {name}", False, str(e).splitlines()[0][:110])

    gc("base", lambda: _cpu_qkv(1, 2, 2, 6, 8),
       lambda a, b, c, mm: _op(a, b, c, mm))
    gc("causal", lambda: _cpu_qkv(1, 2, 2, 6, 8),
       lambda a, b, c, mm: _op(a, b, c, None, 0.0, True, None))
    gc("float-mask ±grad", lambda: _cpu_qkv(1, 2, 2, 6, 8, mask="float"),
       lambda a, b, c, mm: _op(a, b, c, mm), with_mask_grad=True)
    gc("gqa", lambda: _cpu_qkv(1, 4, 2, 6, 8),
       lambda a, b, c, mm: _op(a, b, c, None, 0.0, False, None, gqa=True))
    # 显式 dropout: 掩码必须在 gradcheck **之外**建一次（每次前向重采样
    # 会让数值差分失效，见本文件上面的教训）
    qg, kg, vg, _ = _cpu_qkv(1, 2, 2, 6, 8)
    dm = make_dropout_mask((1, 2, 6, 6), 0.3, "cpu")  # (B,Hq,Sq,Skv)
    try:
        gradcheck(lambda a, b, c: _op(a, b, c, None, 0.3, False, dm),
                  (qg, kg, vg), eps=1e-6, atol=1e-5, rtol=1e-3)
        check("gradcheck: explicit-dropout", True)
    except Exception as e:  # noqa: BLE001
        check("gradcheck: explicit-dropout", False,
              str(e).splitlines()[0][:110])

    # 概率图消费者（双输出里只用 P 的梯度）
    qg, kg, vg, _ = _cpu_qkv(1, 2, 2, 6, 8)
    try:
        gradcheck(lambda a, b, c: _op(a, b, c, None, 0.0, True, None)[1],
                  (qg, kg, vg), eps=1e-6, atol=1e-5, rtol=1e-3)
        check("gradcheck: 概率图消费者 (dP)", True)
    except Exception as e:  # noqa: BLE001
        check("gradcheck: 概率图消费者 (dP)", False,
              str(e).splitlines()[0][:110])

    # ── 6) 错误路径 ──
    qe, ke, ve, me = _mk(1, 4, 4, 32, 32, dev, seed=9, mask="bool",
                         req_grad=False)
    try:
        _op(qe, ke, ve, me, 0.0, True, None)
        check("causal+mask 冲突报错", False, "未抛错")
    except RuntimeError as e:
        check("causal+mask 冲突报错", "Explicit attn_mask" in str(e))
    try:
        q7 = torch.randn(1, 7, 32, 16, device=dev)
        k2x = torch.randn(1, 2, 32, 16, device=dev)
        _op(q7, k2x, k2x, None, 0.0, False, None, gqa=True)
        check("GQA 头不整除报错", False, "未抛错")
    except RuntimeError as e:
        check("GQA 头不整除报错", "divide" in str(e))

    # ── 7) F.sdpa(MATH) 反向消费: 上游走本算子（CPU 键）──
    qc, kc, vc, _ = _mk(1, 4, 4, 48, 48, "cpu", seed=11, req_grad=False)
    ctr_key = ctr_main if profile.dispatch_key == "CPU" else ctr_cpu
    n0 = ctr_key["n"]
    with sdpa_kernel(SDPBackend.MATH):
        f_out = F.scaled_dot_product_attention(qc, kc, vc, is_causal=True)
    hit = ctr_key["n"] > n0
    check("F.sdpa(MATH) 被拦截到本算子（CPU 键）", hit,
          f"+{ctr_key['n'] - n0}")
    ref_f = sdpa_math_reference(qc, kc, vc, None, 0.0, True, None)[0]
    ef = (f_out - ref_f).abs().max().item()
    check("F.sdpa(MATH) 输出 = 参考", ef < 1e-5, f"err={ef:.2e}")

    for ln in results:
        print(ln)
    assert libs_main and (libs_cpu or profile.dispatch_key == "CPU")
    return {"ok": True, "dispatch_key": profile.dispatch_key,
            "impl": profile.default_impl,
            "intercepts": ctr_main["n"] + ctr_cpu["n"],
            "gradcheck": "6 cases pass", "cases": len(results)}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
