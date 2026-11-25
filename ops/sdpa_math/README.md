# sdpa_math-op: `aten::_scaled_dot_product_attention_math`（SDPA math 后端 + 概率图）

> **文档口径**：README 负责入口和当前结论摘要，`REPORT.md` 负责当前验证矩阵，
> `PLATFORM.md` 负责不可迁移的平台绑定，`reports/` 保存带日期的实验记录与原始数据。
> 当前为四个硬件交付（Ascend910、P800、thead-ppu、MLU590）加 CPU ATen 对照；
> CPU 不计作硬件平台。

按 [FlagOS-OP](https://github.com/FLAG-OP/FlagOS-OP) 算子模板开发，
复用同一语义参考、黄金数据、三层测试与 A1 注册链。

## 一页看全

| 项 | 值 |
|---|---|
| 算子 | `aten::_scaled_dot_product_attention_math`（torch SDPA 的 **math 后端**，`CompositeImplicitAutograd`） |
| 语义 | `softmax(QKᵀ·scale + mask)` 双输出：`(out, attn_probs)` · GQA · causal · bool/float mask · dropout 双规则 · fp32 内部（fp64 保持；P800/MLU590 半精度 fast path 见下） |
| 路线 | **A1** aten 拦截（成对注册 `Autograd*` + 纯设备键），旧业务代码零改动 |
| 平台 | **硬件交付**：ascend910（自研 Triton）· p800-kunlunxin（vendor O/LSE + exact-P）· thead-ppu（flash/efficient 分流 + exact-P）· mlu590（Cambricon fused overrideable + exact-P）；**CPU**：ATen 对照 |
| 验证 | kernel 52 组 ×4 profile · 黄金 **175/175** · 原生 131/131（44 bool 跳过）· op 22 项（ascend）/ 41 项（mlu590）· 应用 8-9 项 · FlagOS E2E ✅（p800 · C1；thead 缺 flag_gems，C2 #37；mlu590 无 vLLM runtime，mini consumer 已过、真实引擎待补） |
| NPU 性能（fp16，vs 同为"返回 out+P"的原生 math） | prefill1k D64 **1.22x** · 1k D128 **1.29x** · 2k D128 **2.03x** · GQA **1.67x** · decode **1.17x** · tail100 **1.16x** |
| CPU 性能（fp32，同口径） | 1.05-1.40x（6 形状全过） |
| P800 性能（fp16，同口径） | direct **1.46-2.21x** native；A1 **1.31-2.17x** |
| MLU590 性能（fp16，同口径） | **0.577–0.718x**（**自研更慢，如实保留**；启用边界与优化方向见 `PLATFORM.md §3b`） |

## 快速开始

```bash
python3 example.py ascend910          # 三层一键（kernel/op/framework）
python3 example.py cpu
python3 example.py p800-kunlunxin
python3 example.py thead-ppu
python3 example.py mlu590

python3 test/kernel_level.py ascend910
python3 test/op_level.py ascend910
python3 test/framework_level.py ascend910

# 黄金（CPU 生成一次，跨平台复用）
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl triton   --device npu:0
python3 script/check_accuracy.py --impl p800     --device cuda:1
python3 script/check_accuracy.py --impl thead    --device cuda:6
python3 script/check_accuracy.py --impl mlu590   --device mlu:0
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl native   --device npu:0   # 原生对照

python3 script/bench_perf.py --device npu:0 --register \
  --json-out reports/perf_ascend910.json
python3 script/bench_perf.py --device cuda:1 --register \
  --json-out reports/perf_p800-kunlunxin.json
python3 script/bench_perf.py --device cuda:6 --register \
  --json-out reports/perf_thead-ppu.json
python3 script/bench_perf.py --device mlu:0 --register \
  --json-out reports/perf_mlu590.json
python3 script/bench_f_context.py --device cuda:1 \
  --json-out reports/perf_f_context_p800-kunlunxin.json  # no-P 参考列
python3 script/e2e_flagos.py --device cuda:1 --dtype bfloat16
python3 probes/native_semantics.py ascend910     # native 语义证据表
python3 probes/thead_first_probe.py              # thead 首测证据
