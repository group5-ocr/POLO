"""
glossary_hybrid.json 보강 패처
- 전역 스톱워드/트리거/화이트리스트 엔트리 삽입
- 모든 metric에 norm_range / (있을 경우) min_reasonable 추가
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).parent
GPA  = HERE / "glossary_hybrid.json"

# 2-1) 전역 엔트리 정의
GLOBAL_ENTRIES = [
    {
        "concept_id": "sw.datasets",
        "labels": {"en": "Datasets", "ko": "데이터셋"},
        "aliases": {"en": ["cifar","cifar-10","cifar10","imagenet","voc","coco","lsun","celeba","celeba-hq","svhn","mnist","wmt","kitti","cityscapes","ade20k","nyu"],
                    "ko": ["CIFAR","CIFAR-10","ImageNet","VOC","COCO","LSUN","CelebA","CelebA-HQ","SVHN","MNIST","WMT","KITTI","Cityscapes","ADE20K","NYU"]},
        "regex": {"en": ["\\b(cifar-?10|imagenet|voc|coco|lsun|celeba(?:-hq)?|svhn|mnist|wmt|kitti|cityscapes|ade20k|nyu)\\b"],
                  "ko": ["CIFAR-?10|ImageNet|COCO|VOC|LSUN|CelebA(?:-HQ)?|SVHN|MNIST|WMT|KITTI|Cityscapes|ADE20K|NYU"]},
        "category": "dataset",
        "value_type": "scalar",
        "default_grammar": "none"
    },
    {
        "concept_id": "sw.sections",
        "labels": {"en": "Sections", "ko": "섹션"},
        "aliases": {"en": ["abstract","introduction","related work","background","method","methods","experiment","experiments","results","discussion","conclusion","conclusions","table","figure","appendix","supplementary","dataset","datasets"],
                    "ko": ["초록","서론","관련 연구","배경","방법","실험","결과","논의","결론","표","그림","부록","보충자료","데이터셋"]},
        "regex": {"en": ["\\b(abstract|introduction|related|background|method|methods|experiment|experiments|results|discussion|conclusion|conclusions|table|figure|appendix|supplementary|dataset|datasets)\\b"],
                  "ko": ["초록|서론|관련\\s*연구|배경|방법|실험|결과|논의|결론|표|그림|부록|보충|데이터셋"]},
        "category": "section",
        "value_type": "scalar",
        "default_grammar": "none"
    },
    {
        "concept_id": "ban.method.tokens",
        "labels": {"en": "Forbidden tokens", "ko": "모델명 금지 토큰"},
        "aliases": {"en": ["fr","fid","bleu","cifar","imagenet","lsun","celeba"],
                    "ko": ["fr","fid","bleu","cifar","imagenet","lsun","celeba"]},
        "regex": {"en": ["\\b(fr|fid|bleu|cifar|imagenet|lsun|celeba)\\b"],
                  "ko": ["\\b(fr|fid|bleu|cifar|imagenet|lsun|celeba)\\b"]},
        "category": "forbidden.method_token",
        "value_type": "scalar",
        "default_grammar": "none"
    },
    {
        "concept_id": "method.whitelist",
        "labels": {"en": "Method whitelist", "ko": "모델 화이트리스트"},
        "aliases": {"en": ["UNet","U-Net","ResNet","DenseNet","VGG","Transformer","ViT","Swin","YOLO","RCNN","R-CNN","GAN","VAE","Flow","Mixer","Inception","BERT","GPT"],
                    "ko": ["UNet","U-Net","ResNet","DenseNet","VGG","Transformer","ViT","Swin","YOLO","RCNN","R-CNN","GAN","VAE","Flow","Mixer","Inception","BERT","GPT"]},
        "regex": {"en": ["UNet|U-Net|ResNet|DenseNet|VGG|Transformer|ViT|Swin|YOLO|R-?CNN|GAN|VAE|Flow|Mixer|Inception|BERT|GPT"],
                  "ko": ["UNet|U-Net|ResNet|DenseNet|VGG|Transformer|ViT|Swin|YOLO|R-?CNN|GAN|VAE|Flow|Mixer|Inception|BERT|GPT"]},
        "category": "method.whitelist",
        "value_type": "scalar",
        "default_grammar": "none"
    },
    {
        "concept_id": "viz.hist.trigger",
        "labels": {"en": "Histogram trigger", "ko": "히스토그램 트리거"},
        "aliases": {"en": ["histogram","distribution"], "ko": ["히스토그램","분포"]},
        "regex": {"en": ["\\b(histogram|distribution)\\b"], "ko": ["히스토그램|분포"]},
        "category": "viz.trigger.histogram",
        "value_type": "scalar",
        "default_grammar": "none"
    },
]

# 각 metric의 정규화 범위/최소 유효치
RANGES = {
    # ↑ up-good
    "metric.accuracy":           {"norm_range":[0.0,1.0],   "min_reasonable":0.5},
    "metric.top5_accuracy":      {"norm_range":[0.0,1.0],   "min_reasonable":0.5},
    "metric.precision":          {"norm_range":[0.0,1.0],   "min_reasonable":0.5},
    "metric.recall":             {"norm_range":[0.0,1.0],   "min_reasonable":0.5},
    "metric.specificity":        {"norm_range":[0.0,1.0],   "min_reasonable":0.5},
    "metric.f1":                 {"norm_range":[0.0,1.0],   "min_reasonable":0.5},
    "metric.auroc":              {"norm_range":[0.5,1.0],   "min_reasonable":0.6},
    "metric.auprc":              {"norm_range":[0.0,1.0],   "min_reasonable":0.3},
    "metric.map_detection":      {"norm_range":[0.0,1.0],   "min_reasonable":0.2},
    "metric.average_recall":     {"norm_range":[0.0,1.0]},
    "metric.miou":               {"norm_range":[0.0,1.0],   "min_reasonable":0.3},
    "metric.dice":               {"norm_range":[0.0,1.0]},
    "metric.pq":                 {"norm_range":[0.0,1.0]},
    "metric.iou":                {"norm_range":[0.0,1.0],   "min_reasonable":0.3},
    "metric.mota":               {"norm_range":[0.0,1.0]},
    "metric.motp":               {"norm_range":[0.0,1.0]},
    "metric.hota":               {"norm_range":[0.0,1.0]},
    "metric.idf1":               {"norm_range":[0.0,1.0]},
    "metric.map_ranking":        {"norm_range":[0.0,1.0]},
    "metric.mrr":                {"norm_range":[0.0,1.0]},
    "metric.ndcg_at_k":          {"norm_range":[0.0,1.0]},
    "metric.recall_at_k":        {"norm_range":[0.0,1.0]},
    "metric.precision_at_k":     {"norm_range":[0.0,1.0]},
    "metric.hit_at_k":           {"norm_range":[0.0,1.0]},
    "metric.rouge1":             {"norm_range":[0.0,100.0]},
    "metric.rouge2":             {"norm_range":[0.0,100.0]},
    "metric.rougeL":             {"norm_range":[0.0,100.0]},
    "metric.meteor":             {"norm_range":[0.0,100.0]},
    "metric.bleu":               {"norm_range":[0.0,100.0]},
    "metric.chrf":               {"norm_range":[0.0,100.0]},
    "metric.bertscore":          {"norm_range":[0.0,1.0]},
    "metric.comet":              {"norm_range":[-1.0,1.0]},
    "metric.bleurt":             {"norm_range":[-1.0,1.0]},
    "metric.stoi":               {"norm_range":[0.0,1.0]},
    "metric.pesq":               {"norm_range":[1.0,4.5]},
    "metric.sdr":                {"norm_range":[0.0,20.0]},
    "metric.si_sdr":             {"norm_range":[0.0,20.0]},
    "metric.mos":                {"norm_range":[1.0,5.0]},
    "metric.inception_score":    {"norm_range":[1.0,30.0]},
    "metric.clip_score":         {"norm_range":[0.0,1.0]},
    "metric.r_precision":        {"norm_range":[0.0,1.0]},
    "metric.pass_at_k":          {"norm_range":[0.0,1.0]},
    "metric.code_bleu":          {"norm_range":[0.0,100.0]},
    "metric.humaneval":          {"norm_range":[0.0,100.0]},
    "metric.mbpp":               {"norm_range":[0.0,100.0]},
    "metric.return":             {"norm_range":[-100.0,100.0]},
    "metric.success_rate":       {"norm_range":[0.0,1.0]},
    "metric.spl":                {"norm_range":[0.0,1.0]},
    # ↓ down-good
    "metric.fid":                {"norm_range":[150.0,5.0]},
    "metric.kid":                {"norm_range":[0.3,0.0]},
    "metric.lpips":              {"norm_range":[1.0,0.0]},
    "metric.fvd":                {"norm_range":[1000.0,0.0]},
    "metric.wer":                {"norm_range":[1.0,0.0]},
    "metric.cer":                {"norm_range":[1.0,0.0]},
    # 안전성
    "safety.toxicity":           {"norm_range":[100.0,0.0]},   # 낮을수록 좋음
    "safety.bias_score":         {"norm_range":[3.0,0.0]},     # 값이 작을수록 좋다고 가정
}

def _upsert_global_entries(items: list[dict]) -> None:
    by_id = {it.get("concept_id"): it for it in items}
    for e in GLOBAL_ENTRIES:
        if e["concept_id"] not in by_id:
            items.append(e)

def _patch_metric_ranges(items: list[dict]) -> None:
    for it in items:
        cid = it.get("concept_id")
        if not cid or it.get("category") != "metric":
            continue
        rng = RANGES.get(cid)
        if not rng:
            continue
        # 덮어씌우되, 이미 값이 있으면 유지
        it.setdefault("norm_range", rng["norm_range"])
        if "min_reasonable" in rng:
            it.setdefault("min_reasonable", rng["min_reasonable"])

def main():
    if not GPA.exists():
        raise SystemExit(f"not found: {GPA}")
    items = json.loads(GPA.read_text(encoding="utf-8"))
    _upsert_global_entries(items)
    _patch_metric_ranges(items)
    GPA.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ glossary_hybrid.json patched:", GPA)

if __name__ == "__main__":
    main()