import torch
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from autoweld_vision.evaluation.metrics import AdvancedIndustrialMetrics


class DomainShiftEvaluator:
    """
    Priority 3.1: Domain Shift Evaluation.
    Measures how models generalize across different industrial datasets.
    """

    def __init__(self, models: Dict[str, Any]):
        self.models = models

    def evaluate_cross_dataset(
        self, source_name: str, target_dataloader: torch.utils.data.DataLoader
    ):
        """Tests a model trained on source_name on a new target_dataset."""
        results = {}
        for name, model in self.models.items():
            model.eval()
            scores = []
            labels = []

            with torch.no_grad():
                for batch in target_dataloader:
                    images = batch["image"]
                    target_labels = batch["label"]

                    output = model(images)
                    scores.extend(output["score"].cpu().numpy())
                    labels.extend(target_labels.cpu().numpy())

            report = AdvancedIndustrialMetrics.generate_full_report(
                np.array(labels), np.array(scores)
            )
            results[name] = report

        return results


class FailureAnalyzer:
    """
    Priority 4.1: Systematic Failure Analysis.
    Visualizes and categorizes model errors (False Positives/Negatives).
    """

    def __init__(self, output_dir: str = "failures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_failures(self, model, dataloader, threshold: float, k: int = 20):
        fps = []  # False Positives
        fns = []  # False Negatives

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"]
                labels = batch["label"]
                paths = batch["path"]

                outputs = model(images)
                scores = outputs["score"].squeeze()
                preds = (scores > threshold).int()

                for i in range(len(labels)):
                    if preds[i] == 1 and labels[i] == 0:
                        fps.append(
                            {
                                "image": images[i],
                                "score": scores[i].item(),
                                "path": paths[i],
                            }
                        )
                    elif preds[i] == 0 and labels[i] == 1:
                        fns.append(
                            {
                                "image": images[i],
                                "score": scores[i].item(),
                                "path": paths[i],
                            }
                        )

        # Sort by most 'confident' errors
        fps = sorted(fps, key=lambda x: x["score"], reverse=True)[:k]
        fns = sorted(fns, key=lambda x: x["score"])[:k]

        self._visualize_grid(fps, "False_Positives")
        self._visualize_grid(fns, "False_Negatives")

        return len(fps), len(fns)

    def _visualize_grid(self, samples: List[Dict], title: str):
        if not samples:
            return
        n = len(samples)
        cols = 5
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for i, sample in enumerate(samples):
            img = sample["image"].permute(1, 2, 0).cpu().numpy()
            img = img * 0.229 + 0.485  # Denormalize (approx)
            axes[i].imshow(np.clip(img, 0, 1))
            axes[i].set_title(f"Score: {sample['score']:.3f}")
            axes[i].axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{title}.png"))
        plt.close()


if __name__ == "__main__":
    import os
    from autoweld_vision.data.mvtec import MVTecADLoader
    from autoweld_vision.models import PatchCoreModel
    from torchvision import transforms as T
    from torch.utils.data import DataLoader

    print("🚀 Starting Benchmark Smoke Test...")

    # 1. Setup Data
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    loader = MVTecADLoader(
        root="./datasets/mvtec", category="bottle", split="test", transform=transform
    )
    dataloader = DataLoader(loader, batch_size=2)

    # 2. Setup Model
    model = PatchCoreModel()

    # 3. Run Evaluation
    evaluator = DomainShiftEvaluator({"patchcore": model})
    results = evaluator.evaluate_cross_dataset("source", dataloader)

    print("📈 Benchmark Results:")
    for m, r in results.items():
        print(f"Model: {m}")
        for metric, val in r.items():
            print(f"  {metric}: {val}")

    print("✓ Benchmark smoke test completed successfully.")
