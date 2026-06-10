"""DCR evaluation for ablation_study."""

import json
import os

import pandas as pd

from ablation_study.scripts.evaluation.dataloader import TabularDataset
from ablation_study.scripts.evaluation.progress_reporter import NullProgressReporter


def evaluate_dcr(args, data_name, variant_slug, synthetic_path, output_dir, reporter=None, synthetic_frame=None, evaluation_cache=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    verbose_enabled = bool(getattr(args, "verbose_eval", False))
    epoch_bar = reporter.create_epoch_bar(4, desc=f"{variant_slug}-dcr", enabled=verbose_enabled)
    detail_bar = reporter.create_detail_bar(4, desc=f"{variant_slug}-dcr-detail", enabled=verbose_enabled)

    try:
        dataset = TabularDataset(
            data_name,
            synthetic_path,
            data_dir=args.data_dir,
            original_test=False,
            synthetic_frame=synthetic_frame,
            evaluation_cache=evaluation_cache,
        )
        X_train, X_test, y_train, y_test, _, _, _ = dataset.get_data(test_num=args.test_num if args.test else None)
        synt_data = pd.concat([X_train, y_train], axis=1)
        real_data = pd.concat([X_test, y_test], axis=1)
        detail_bar.update(1)
        detail_bar.set_postfix({"stage": "load-data", "data": data_name}, refresh=True)
        epoch_bar.update(1)
        epoch_bar.set_postfix({"stage": "load-data", "data": data_name}, refresh=True)

        with open(os.path.join(args.data_dir, "cols_info", f"{data_name}_metadata.json"), "r", encoding="utf-8") as file:
            meta_data = json.load(file)["tables"]["table"]["columns"]
        meta_cols = set(meta_data.keys())
        data_cols = [c for c in synt_data.columns if c in meta_cols and c in real_data.columns]
        synt_data = synt_data[data_cols]
        real_data = real_data[data_cols]
        meta_data_filtered = {c: meta_data[c] for c in data_cols}
        detail_bar.update(1)
        detail_bar.set_postfix({"stage": "load-meta", "cols": len(data_cols)}, refresh=True)
        epoch_bar.update(1)
        epoch_bar.set_postfix({"stage": "load-meta", "cols": len(data_cols)}, refresh=True)

        from prediction.scripts.dcr_evaluation import calculate_dcr_score

        score = calculate_dcr_score(real_data, synt_data, meta_data_filtered, device=getattr(args, "device_dcr", "cpu"))
        detail_bar.update(1)
        detail_bar.set_postfix({"stage": "compute", "data": data_name}, refresh=True)
        epoch_bar.update(1)
        epoch_bar.set_postfix({"stage": "compute", "data": data_name}, refresh=True)

        dcr_dir = os.path.join(output_dir, "DCR")
        os.makedirs(dcr_dir, exist_ok=True)
        pd.DataFrame({"data_name": [data_name], variant_slug: [score]}).to_csv(
            os.path.join(dcr_dir, "DCR_scores.csv"), index=False
        )
        detail_bar.update(1)
        detail_bar.set_postfix({"stage": "save", "data": data_name}, refresh=True)
        epoch_bar.update(1)
        epoch_bar.set_postfix({"stage": "save", "data": data_name}, refresh=True)
    finally:
        detail_bar.close()
        epoch_bar.close()

    reporter.ok(f"[OK] metric=DCR variant={variant_slug} data={data_name}")
