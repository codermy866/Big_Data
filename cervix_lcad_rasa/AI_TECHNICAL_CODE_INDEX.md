# MOSAIC Technical Code Index for AI Review

This index points to the minimal implementation files that explain the core
technical content of MOSAIC/LCAD-RASA. It intentionally excludes generated
figures, patient images, prediction tables, and other data artifacts.

## Core Method

- `src/models_publishable/lcad_rasa_model.py`
  - Main publishable MOSAIC/LCAD-RASA model.
  - Implements OCT, colposcopy, fused-visual, and structured-clinical branches;
    multimodal fusion; report decoding; risk prediction; and section alignment.

- `src/models/lcad_rasa.py`
  - Lightweight reference implementation of the LCAD-RASA architecture.
  - Useful for quickly understanding the basic encoder, report decoder,
    risk head, RASA alignment loss, and LCAD distillation loss.

- `src/models/section_alignment.py`
  - Section-level semantic alignment utilities.
  - Encodes the report-anchor idea: OCT evidence should align with OCT report
    sections, colposcopy evidence with colposcopy sections, and clinical
    variables with clinical-context sections.

## Weak-Oracle Report Supervision

- `src/distillation/pseudo_report_schema.py`
  - Structured schema for pseudo-report sections.
  - Defines the report sections used as weak semantic supervision.

- `src/distillation/report_topic_distiller.py`
  - Topic/semantic distillation logic over report content.
  - Used to extract structured report priors instead of treating reports as
    unstructured free text.

- `src/distillation/qc.py`
  - Quality-control filters for generated or weak reports.
  - Important for bounding LLM hallucination and preventing low-quality
    pseudo-reports from dominating training.

- `src/data/report_supervision.py`
  - Dataset-level report-supervision handling.
  - Connects real reports, pseudo-reports, QC flags, and training weights.

## Retrieval-Calibrated Semantic Prior

- `src/retrieval/semantic_bank.py`
  - Train-only report-derived semantic memory bank.
  - Builds section-level knowledge entities and retrieves semantic priors
    without using validation or test cases to construct memory.

- `scripts/46_build_cervical_semantic_retrieval.py`
  - Pipeline entry point for building cervical semantic retrieval artifacts.

- `scripts/49_analyze_kra_semantic_fusion.py`
  - Analysis of retrieval-calibrated semantic fusion.
  - Connects retrieved priors to validation-calibrated fusion behavior.

## Training, Losses, and Distillation

- `src/training/publishable_dataset.py`
  - Publishable dataset wrapper for multimodal embeddings, structured clinical
    variables, reports, labels, and pseudo-report weights.

- `src/training/losses.py`
  - Shared loss components used by the training pipeline.

- `src/training/trainer.py`
  - Generic training loop utilities.

- `scripts/19_train_publishable_lcad_rasa.py`
  - Main publishable training entry point for LCAD-RASA variants.

- `scripts/43_run_contrastive_teacher_distillation.py`
  - Contrastive-teacher distillation audit.
  - This is intentionally audit-gated: the distilled setting is promotable only
    if it improves locked reference metrics under the predefined validation
    and test protocol.

## Evaluation and Audit

- `scripts/37_run_jbd_theme1_alignment_experiments.py`
  - Theme-1 alignment experiments.
  - Computes section-level retrieval/alignment evidence such as recall, MRR,
    and positive-negative semantic gaps.

- `src/evaluation_publishable/section_consistency.py`
  - Report-section consistency metrics.

- `src/evaluation_publishable/perturbation_metrics.py`
  - Modality perturbation metrics for checking whether report sections and risk
    scores respond to missing or masked evidence in an auditable way.

## Manuscript-Level Technical Summary

- `CORE_TECHNICAL_IDEAS_AND_INNOVATIONS.md`
  - Human-readable summary of the implemented technical ideas and claim
    boundaries.
  - Best starting point before reading the code files above.
