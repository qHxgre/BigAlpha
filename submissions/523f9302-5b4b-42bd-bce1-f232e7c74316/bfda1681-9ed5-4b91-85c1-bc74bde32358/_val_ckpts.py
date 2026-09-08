# -*- coding: utf-8 -*-
"""独立验证脚本 —— 训练结束后自行运行, 评估所有 epoch 检查点并选出最佳。

用法: python _val_ckpts.py

会自动:
  1. 发现 transformer_model.json.ep* 检查点
  2. 加载缓存的标准化统计 (transformer_stats.json)
  3. 在验证集上评估每个检查点
  4. 将最佳检查点保存为 transformer_model.json.best

支持两种模式 (由 config.MODE 控制):
  - local:  本地 parquet 数据, 使用 compute_val_metrics 计算 IC/IR/多空Sharpe
  - online: BigQuant 平台 dai 数据, 使用 M.bigalpha_eval._latest 获取在线平台指标
            指标来源: result['factor_analyze']['ic_mean']
                      result['factor_analyze']['ic_ir']
                      result['factor_analyze']['sharpe_ratio']
                      result['factor_analyze']['stress_ic_ir']
"""
import os, sys, json, gc, glob
import numpy as np, pandas as pd, torch, structlog

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config
from train import StockTransformer, StockTransformerV2, StockTransformerV3, build_dataset, load_model, save_model, compute_val_metrics

structlog.configure(processors=[structlog.dev.ConsoleRenderer()], cache_logger_on_first_use=True)
logger = structlog.get_logger()


def discover_checkpoints():
    """按 ep 编号排序发现所有检查点文件。"""
    pattern = config.MODEL_PATH + ".ep*"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未找到检查点文件: {pattern}")
    logger.info(f"发现 {len(paths)} 个检查点", paths=[os.path.basename(p) for p in paths])
    return paths


def _load_stats(ckpt_paths):
    """加载标准化统计: 优先从 transformer_stats.json, 回退到首个检查点。"""
    if os.path.exists(config.STATS_PATH):
        with open(config.STATS_PATH, "r") as f:
            stats_payload = json.load(f)
        stats = (np.array(stats_payload["mean"], dtype=np.float32),
                 np.array(stats_payload["std"], dtype=np.float32))
        logger.info("从 stats 文件加载标准化统计", path=config.STATS_PATH)
    else:
        first_ckpt = load_model(ckpt_paths[0], map_location="cpu")
        stats = (np.asarray(first_ckpt["mean"], np.float32),
                 np.asarray(first_ckpt["std"], np.float32))
        logger.info("从首个检查点加载标准化统计")
    return stats


def _eval_local(X_val, y_val, keys_val, device, ckpt_paths, best_path):
    """本地模式验证: 使用 compute_val_metrics 计算 IC/IR/多空Sharpe。"""
    best_metrics = {"ic_mean": float("-inf"), "ic_ir": float("nan"),
                    "long_short_sharpe": float("nan")}
    best_ep = 0

    for ckpt_path in ckpt_paths:
        ckpt = load_model(ckpt_path, map_location="cpu")
        model_cls_name = ckpt.get("model_class", "StockTransformer")
        if model_cls_name == "StockTransformerV3":
            model = StockTransformerV3(**ckpt["model_cfg"]).to(device)
        elif model_cls_name == "StockTransformerV2":
            model = StockTransformerV2(**ckpt["model_cfg"]).to(device)
        else:
            model = StockTransformer(**ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        preds = []
        X_val_t = torch.from_numpy(X_val)
        with torch.no_grad():
            for i in range(0, len(X_val_t), config.BATCH):
                xb = X_val_t[i:i + config.BATCH].to(device)
                preds.append(model(xb).cpu().numpy())
        y_pred = np.concatenate(preds)

        em = compute_val_metrics(y_pred, y_val, keys_val)
        ep_num = ckpt.get("epoch", 0)
        marker = ""
        if em["ic_mean"] > best_metrics["ic_mean"]:
            best_metrics = em; best_ep = ep_num; marker = " *"
            ckpt["val_metrics"] = em; ckpt["best_epoch"] = best_ep
            save_model(ckpt, best_path)
        logger.info(f"  ep {ep_num:02d}  ic={em['ic_mean']}  ir={em['ic_ir']}  "
                    f"sharpe={em['long_short_sharpe']}{marker}")
        del model; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    logger.info("最佳", epoch=best_ep, ic_mean=best_metrics["ic_mean"],
                ic_ir=best_metrics["ic_ir"], ls_sharpe=best_metrics["long_short_sharpe"])
    return best_path


def _eval_online(X_val, keys_val, device, ckpt_paths, best_path):
    """在线模式验证: 使用 M.bigalpha_eval._latest 获取平台指标。"""
    import dai
    from bigmodule import M

    # 预取 instrument 参考表 (与 predict.ipynb 在线流程一致)
    val_start, val_end = config.VAL_START, config.VAL_END
    stk = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                    filters={"date": [val_start, val_end]}).df()

    best_metrics = {"ic_mean": float("-inf"), "ic_ir": float("nan"),
                    "sharpe_ratio": float("nan"), "stress_ic_ir": float("nan")}
    best_ep = 0

    for ckpt_path in ckpt_paths:
        ckpt = load_model(ckpt_path, map_location="cpu")
        model_cls_name = ckpt.get("model_class", "StockTransformer")
        if model_cls_name == "StockTransformerV3":
            model = StockTransformerV3(**ckpt["model_cfg"]).to(device)
        elif model_cls_name == "StockTransformerV2":
            model = StockTransformerV2(**ckpt["model_cfg"]).to(device)
        else:
            model = StockTransformer(**ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        preds = []
        X_val_t = torch.from_numpy(X_val)
        with torch.no_grad():
            for i in range(0, len(X_val_t), config.BATCH):
                xb = X_val_t[i:i + config.BATCH].to(device)
                preds.append(model(xb).cpu().numpy())
        y_pred = np.concatenate(preds)

        # 构建 score_data: date + instrument + score
        idx_df = keys_val.copy()
        idx_df["score"] = y_pred.astype(np.float64)

        # 与 instrument 参考表 inner join, 清洗 (与 predict.ipynb 在线流程一致)
        result_df = (pd.merge(idx_df, stk, on=["date", "instrument"], how="inner")
                      .replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
                      .drop_duplicates(["date", "instrument"])[["date", "instrument", "score"]]
                      .reset_index(drop=True))

        eval_result = M.bigalpha_eval._latest(factor_data=result_df)
        fa = eval_result["factor_analyze"]
        em = {
            "ic_mean": fa["ic_mean"],
            "ic_ir": fa["ic_ir"],
            "sharpe_ratio": fa["sharpe_ratio"],
            "stress_ic_ir": fa["stress_ic_ir"],
        }

        ep_num = ckpt.get("epoch", 0)
        marker = ""
        if em["ic_mean"] > best_metrics["ic_mean"]:
            best_metrics = em; best_ep = ep_num; marker = " *"
            ckpt["val_metrics"] = em; ckpt["best_epoch"] = best_ep
            save_model(ckpt, best_path)
        logger.info(f"  ep {ep_num:02d}  ic={em['ic_mean']:.6f}  ir={em['ic_ir']:.4f}  "
                    f"sharpe={em['sharpe_ratio']:.4f}  stress_ir={em['stress_ic_ir']:.4f}{marker}")
        del model; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    logger.info("最佳", epoch=best_ep, ic_mean=best_metrics["ic_mean"],
                ic_ir=best_metrics["ic_ir"], sharpe=best_metrics["sharpe_ratio"],
                stress_ic_ir=best_metrics["stress_ic_ir"])
    return best_path


def main():
    val_start = config.VAL_START
    val_end = config.VAL_END
    best_path = config.MODEL_PATH + ".best"

    ckpt_paths = discover_checkpoints()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = _load_stats(ckpt_paths)

    tables = config.FREQ_TABLES

    # 获取 instrument 列表
    instruments = config.pool(val_start, val_end)
    if config.MODE == "local":
        instruments = instruments[:config.MAX_TRAIN_INSTRUMENTS]

    # 本地模式使用 "train" (需 label 计算指标), 在线模式使用 "infer" (与 predict.ipynb 一致)
    mode = "train" if config.MODE == "local" else "infer"
    logger.info("加载验证集", start=val_start, end=val_end,
                n_inst=len(instruments), mode=config.MODE, build_mode=mode)
    X_val, y_val, keys_val, _ = build_dataset(
        tables, val_start, val_end, mode, instruments, stats=stats)

    if config.MODE == "local":
        _eval_local(X_val, y_val, keys_val, device, ckpt_paths, best_path)
    else:
        _eval_online(X_val, keys_val, device, ckpt_paths, best_path)

    logger.info("最佳检查点已保存", path=best_path)


if __name__ == "__main__":
    main()
