"""Map each manuscript figure stem to a Seaborn gallery reference template."""

from __future__ import annotations

# Keys match PNG filenames under experiments/JBD_2026/Seaborn/
SEABORN_GALLERY_TEMPLATES: dict[str, dict[str, str]] = {
    "Figure2_centre_supervision_catplot": {
        "gallery": "Scatterplot with varying point sizes and hues.png",
        "seaborn_api": "scatterplot + size encoding",
        "code": "jbd_figures_seaborn.fig02_centre_supervision",
    },
    "Figure_mosaic_performance_summary": {
        "gallery": "Joint kernel density estimate.png + Grouped violinplots with split violins.png",
        "seaborn_api": "JointGrid + violinplot composite",
        "code": "scripts/50_generate_kra_semantic_fusion_figures.make_summary_figure",
    },
    "Figure_mosaic_metrics_heatmap": {
        "gallery": "Dot plot with several variables.png",
        "seaborn_api": "horizontal lollipop / scatter profile",
        "code": "scripts/50_generate_kra_semantic_fusion_figures.make_metric_lollipop",
    },
    "Figure_external_baselines_auc_forest": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "horizontal_lollipop + errorbar",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_external_baselines",
    },
    "Figure_external_baselines_metric_dotplot": {
        "gallery": "Dot plot with several variables.png",
        "seaborn_api": "FacetGrid + stripplot",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_external_baselines",
    },
    "Figure_external_baselines_paired_delta_auc": {
        "gallery": "Regression fit over a strip plot.png",
        "seaborn_api": "horizontal_lollipop_pvals",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_external_baselines",
    },
    "Figure_theme1_pseudo_report_source_comparison": {
        "gallery": "Dot plot with several variables.png",
        "seaborn_api": "FacetGrid + stripplot (no connecting lines)",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_theme1_pseudo",
    },
    "Figure_theme1_report_supervision_scarcity_curve": {
        "gallery": "Annotated heatmaps.png",
        "seaborn_api": "scarcity_heatmap",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_scarcity_curve",
    },
    "Figure_theme1_alignment_retrieval_mrr": {
        "gallery": "Line plots on multiple facets.png",
        "seaborn_api": "FacetGrid scatter (no line)",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_theme1_alignment",
    },
    "Figure3_modality_perturbation_heatmap": {
        "gallery": "Discovering structure in heatmap data.png",
        "seaborn_api": "clustermap_figure",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_perturbation",
    },
    "Figure3_modality_perturbation_lineplot": {
        "gallery": "Horizontal, unfilled violinplots.png",
        "seaborn_api": "FacetGrid + stripplot orient=h",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_perturbation",
    },
    "Figure3_risk_delta_stripplot": {
        "gallery": "Scatterplot with continuous hues and sizes.png",
        "seaborn_api": "scatter + sequential hue",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_perturbation",
    },
    "Figure_theme1_perturbation_sensitivity_matrix": {
        "gallery": "Discovering structure in heatmap data.png",
        "seaborn_api": "clustermap_figure diverging",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_theme1_perturbation_matrix",
    },
    "fig_loco_heatmap": {
        "gallery": "Scatterplot with categorical variables.png",
        "seaborn_api": "relplot scatter hue/style",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_loco",
    },
    "Figure4_loco_forest_catplot": {
        "gallery": "Scatterplot with categorical variables.png",
        "seaborn_api": "same as fig_loco_heatmap",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_loco",
    },
    "Figure_main_AUC_pointplot": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "horizontal_lollipop_pvals",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_main_comparison",
    },
    "Figure_main_metrics_heatmap": {
        "gallery": "Plotting a diagonal correlation matrix.png",
        "seaborn_api": "diagonal_heatmap",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_main_comparison",
    },
    "Figure_main_auc_f1_scatter": {
        "gallery": "Linear regression with marginal distributions.png",
        "seaborn_api": "joint_scatter_marginals",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_main_comparison",
    },
    "fig_rasa_lambda_lineplot": {
        "gallery": "Conditional means with observations.png",
        "seaborn_api": "lambda_sweep_dumbbell",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_ablations",
    },
    "fig_modality_ablation_stripplot": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "boxenplot + stripplot",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_ablations",
    },
    "fig_rasa_component_boxenplot": {
        "gallery": "Grouped boxplots.png",
        "seaborn_api": "boxenplot + stripplot",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_ablations",
    },
    "fig_lcad_qc_ablation_barplot": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "horizontal_box_strip",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_ablations",
    },
    "SupplementaryFigure_S1_masking_validation": {
        "gallery": "Grouped violinplots with split violins.png",
        "seaborn_api": "grouped_violin_strip",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_robustness",
    },
    "SupplementaryFigure_S3_multiseed": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "horizontal_lollipop_pvals",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_robustness",
    },
    "P1_stage1_quality_heatmap": {
        "gallery": "Annotated heatmaps.png",
        "seaborn_api": "sns.heatmap annotated",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_quality",
    },
    "P2_stage1_quality_risk_bars": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "boxenplot + stripplot orient=h",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_quality",
    },
    "P3_stage1_latency_support_scatter": {
        "gallery": "Scatterplot with varying point sizes and hues.png",
        "seaborn_api": "bubble scatter",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_quality",
    },
    "P4_stage1_generation_reliability": {
        "gallery": "Annotated heatmaps.png",
        "seaborn_api": "proportion_heatmap",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_reliability",
    },
    "P5_stage2_macro_mrr": {
        "gallery": "Horizontal boxplot with observations.png",
        "seaborn_api": "lollipop forest",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_alignment",
    },
    "P6_stage2_section_mrr": {
        "gallery": "Line plots on multiple facets.png",
        "seaborn_api": "FacetGrid scatter",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_alignment",
    },
    "P7_stage3_scarcity_auc": {
        "gallery": "Annotated heatmaps.png",
        "seaborn_api": "scarcity_heatmap",
        "code": "scripts/39_generate_llm_api_paper_ready_outputs._plot_scarcity",
    },
    "P8_llm_provider_comparison_heatmap": {
        "gallery": "Discovering structure in heatmap data.png",
        "seaborn_api": "clustermap_figure",
        "code": "scripts/56_redraw_individual_novel_figures.redraw_api_p8_clustermap",
    },
    "Figure1_mosaic_overview": {
        "gallery": "N/A (schematic)",
        "seaborn_api": "matplotlib schematic",
        "code": "scripts/51_generate_mosaic_figure1_overview",
    },
}
