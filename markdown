def build_subsidiary_markdowns(df: pd.DataFrame, all_weeks: List[pd.Timestamp]) -> None:
    if not all_weeks:
        return

    max_dt = df["Response_Date__UTC_"].dropna().max()
    if pd.isna(max_dt):
        max_dt = pd.Timestamp.today().normalize()
    this_week = _pick_completed_week_end(all_weeks, max_dt)

    subsidiaries = sorted([s for s in df["Subsidiary"].dropna().unique()])

    for sub in subsidiaries:
        # if sub not in ["SEG","SEA","SEC"]:
        #     continue
        sub_df = df[df["Subsidiary"] == sub].copy()
        if sub_df.empty:
            md = f"# {sub} NPS Report\n데이터가 없습니다.\n\n" + sm_local("")
            out_path = os.path.join(SUB_MD_DIR, f"{sub}.md")
            safe_path = _safe_filename(out_path)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(md)
            continue

        # ---- 1) 최근 6주 텍스트 라인 ----
        weeks_span = _recent_weeks_up_to(all_weeks, this_week, k=6)
        sub_series_all = (sub_df.groupby("week_end")["Likelihood_to_Recommend"]
                                .apply(nps_from_scores)
                                .reindex(weeks_span))
        n_points = len(weeks_span)
        parts = []
        for i, wk in enumerate(weeks_span):
            step_back = (n_points - 1) - i
            label = f"W-{step_back}" if step_back > 0 else "W "
            val = np_round(sub_series_all.iloc[i])
            parts.append(f"{label}({val})")
        trend_line = "- " + " -> ".join(parts) if parts else "- 데이터 없음"

        # ---- Impact influence helper (이번주 기준) ----
        def impact_influence(tp: str, factor_col: str) -> pd.DataFrame:
            part = sub_df[sub_df["Sub_Service_Type"].str.lower() == tp.lower()].copy()
            part = part[part["week_end"] == this_week]
            if part.empty:
                return pd.DataFrame(columns=["ImpactFactor","NPS","Count","Influence"])

            part["ImpactFactor"] = part[factor_col]
            part.loc[part["ImpactFactor"].isna() | (part["ImpactFactor"] == ""), "ImpactFactor"] = "(Unspecified)"

            overall_nps = nps_from_scores(part["Likelihood_to_Recommend"])
            total_n = part["Likelihood_to_Recommend"].notna().sum()

            g_nps = part.groupby("ImpactFactor")["Likelihood_to_Recommend"].apply(nps_from_scores)
            g_cnt = part.groupby("ImpactFactor")["Likelihood_to_Recommend"].apply(lambda s: s.notna().sum())
            out = pd.DataFrame({"ImpactFactor": g_nps.index, "NPS": g_nps.values, "Count": g_cnt.values})

            if total_n > 0:
                out["Influence"] = (out["Count"].astype(float) / float(total_n)) * (out["NPS"] - overall_nps)
            else:
                out["Influence"] = np.nan

            out = out.sort_values(by=["Influence","ImpactFactor"], ascending=[True, True]).reset_index(drop=True)
            out["NPS"] = out["NPS"].apply(np_round)
            out["Influence"] = out["Influence"].apply(np_round)
            return out

        # call_imp = impact_influence("Call","Call_Impact_Factor")
        # chat_imp = impact_influence("Chat","Chat_Impact_Factor")
        CI_imp = impact_influence("CI","CS_Repair_Impact_Factor")
        PS_imp = impact_influence("PS","CS_Repair_Impact_Factor")
        IH_imp = impact_influence("IH","CS_Repair_Impact_Factor")

        # ---- 2) CI NPS 분석 결과 ----
        ci_all = sub_df[sub_df["Sub_Service_Type"].str.lower() == "ci"].copy()
        pos_ci = ci_all[ci_all["Likelihood_to_Recommend"] >= 9].copy()
        neg_ci = ci_all[ci_all["Likelihood_to_Recommend"] <= 6].copy()

        pos_samples = _sample_texts(pos_ci, 5, f"{sub}-ci-pos")
        neg_samples = _sample_texts(neg_ci, 5, f"{sub}-ci-neg")

        pos_summary = nps_sum(" ".join(pos_samples)) if pos_samples else "요약"
        neg_summary = nps_sum(" ".join(neg_samples)) if neg_samples else "요약"

        ci_pairs_lines = []
        if not ci_all.empty:
            for _, r in ci_all.iterrows():
                reason = clean_reason_str(r.get("Comment", ""))
                if not reason or reason.lower() == "no reason":
                    continue
                impact = r.get("CS_Repair_Impact_Factor", "")
                impact = impact if impact and str(impact).strip() else "(Unspecified)"
                ci_pairs_lines.append(f"{impact}::{reason}")
        # ci_classification = Classification("\n".join(ci_pairs_lines)[:100000]) if ci_pairs_lines else "분류할 데이터가 없습니다."

        # ---- 3) PS NPS 분석 결과 ----
        ps_all = sub_df[sub_df["Sub_Service_Type"].str.lower() == "ps"].copy()
        pos_ps = ps_all[ps_all["Likelihood_to_Recommend"] >= 9].copy()
        neg_ps = ps_all[ps_all["Likelihood_to_Recommend"] <= 6].copy()

        pos_ps_samples = _sample_texts(pos_ps, 5, f"{sub}-ps-pos")
        neg_ps_samples = _sample_texts(neg_ps, 5, f"{sub}-ps-neg")

        pos_ps_summary = nps_sum(" ".join(pos_ps_samples)) if pos_ps_samples else "요약"
        neg_ps_summary = nps_sum(" ".join(neg_ps_samples)) if neg_ps_samples else "요약"

        ps_pairs_lines = []
        if not ps_all.empty:
            for _, r in ps_all.iterrows():
                reason = clean_reason_str(r.get("Comment", ""))
                if not reason or reason.lower() == "no reason":
                    continue
                impact = r.get("CS_Repair_Impact_Factor", "")
                impact = impact if impact and str(impact).strip() else "(Unspecified)"
                ps_pairs_lines.append(f"{impact}::{reason}")
        # ps_classification = Classification_PS("\n".join(ps_pairs_lines)) if ps_pairs_lines else "분류할 데이터가 없습니다."

        # ---- 4) IH NPS 분석 결과 ----
        ih_all = sub_df[sub_df["Sub_Service_Type"].str.lower() == "ih"].copy()
        pos_ih = ih_all[ih_all["Likelihood_to_Recommend"] >= 9].copy()
        neg_ih = ih_all[ih_all["Likelihood_to_Recommend"] <= 6].copy()

        pos_ih_samples = _sample_texts(pos_ih, 5, f"{sub}-ih-pos")
        neg_ih_samples = _sample_texts(neg_ih, 5, f"{sub}-ih-neg")

        pos_ih_summary = nps_sum(" ".join(pos_ih_samples)) if pos_ih_samples else "요약"
        neg_ih_summary = nps_sum(" ".join(neg_ih_samples)) if neg_ih_samples else "요약"

        ih_pairs_lines = []
        if not ih_all.empty:
            for _, r in ih_all.iterrows():
                reason = clean_reason_str(r.get("Comment", ""))
                if not reason or reason.lower() == "no reason":
                    continue
                impact = r.get("CS_Repair_Impact_Factor", "")
                impact = impact if impact and str(impact).strip() else "(Unspecified)"
                ih_pairs_lines.append(f"{impact}::{reason}")
        # ih_classification = Classification_IH("\n".join(ih_pairs_lines)) if ih_pairs_lines else "분류할 데이터가 없습니다."

        # ---- Build markdown ----
        md_lines = [] 
        md_lines.append(f"# {sub} NPS Report")
        md_lines.append("## 1) NPS trend")
        md_lines.append(trend_line)
        md_lines.append("")
        md_lines.append("## 2) CI NPS analysis results")
        md_lines.append("### Positive Summary")
        md_lines.append(pos_summary)
        if pos_samples:
            for s in pos_samples:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("_해당 리뷰 없음_")
        md_lines.append("")
        md_lines.append("### Negative Summary")
        md_lines.append(neg_summary)
        if neg_samples:
            for s in neg_samples:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("_해당 리뷰 없음_")
        md_lines.append("")
        md_lines.append("### Impact Factor Table (CI)")
        if CI_imp.empty:
            md_lines.append("_데이터 없음_")
        else:
            md_lines.append("| Impact Factor | NPS | Count | Influence |")
            md_lines.append("|---|---:|---:|---:|")
            for _, r in CI_imp.iterrows():
                md_lines.append(f"| {r['ImpactFactor']} | {r['NPS']} | {int(r['Count']) if not pd.isna(r['Count']) else 0} | {r['Influence']} |")
        md_lines.append("")
        # md_lines.append("### Detailed classification results (CI)")
        # md_lines.append(ci_classification)
        # md_lines.append("")
        md_lines.append("## 3) PS NPS analysis results")
        md_lines.append("### Positive Summary")
        md_lines.append(pos_ps_summary)
        if pos_ps_samples:
            for s in pos_ps_samples:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("_해당 리뷰 없음_")
        md_lines.append("")
        md_lines.append("### Negative Summary")
        md_lines.append(neg_ps_summary)
        if neg_ps_samples:
            for s in neg_ps_samples:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("_해당 리뷰 없음_")
        md_lines.append("")
        md_lines.append("### Impact Factor Table (PS)")
        if PS_imp.empty:
            md_lines.append("_데이터 없음_")
        else:
            md_lines.append("| Impact Factor | NPS | Count | Influence |")
            md_lines.append("|---|---:|---:|---:|")
            for _, r in PS_imp.iterrows():
                md_lines.append(f"| {r['ImpactFactor']} | {r['NPS']} | {int(r['Count']) if not pd.isna(r['Count']) else 0} | {r['Influence']} |")
        md_lines.append("")
        # md_lines.append("### Detailed classification results (PS)")
        # md_lines.append(ps_classification)
        # md_lines.append("")
        md_lines.append("## 4) IH NPS analysis results")
        md_lines.append("### Positive Summary")
        md_lines.append(pos_ih_summary)
        if pos_ih_samples:
            for s in pos_ih_samples:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("_해당 리뷰 없음_")
        md_lines.append("")
        md_lines.append("### Negative Summary")
        md_lines.append(neg_ih_summary)
        if neg_ih_samples:
            for s in neg_ih_samples:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("_해당 리뷰 없음_")
        md_lines.append("")
        md_lines.append("### Impact Factor Table (IH)")
        if IH_imp.empty:
            md_lines.append("_데이터 없음_")
        else:
            md_lines.append("| Impact Factor | NPS | Count | Influence |")
            md_lines.append("|---|---:|---:|---:|")
            for _, r in IH_imp.iterrows():
                md_lines.append(f"| {r['ImpactFactor']} | {r['NPS']} | {int(r['Count']) if not pd.isna(r['Count']) else 0} | {r['Influence']} |")
        md_lines.append("")
        # md_lines.append("### Detailed classification results (IH)")
        # md_lines.append(ih_classification)
        # md_lines.append("")

        md = "\n".join(md_lines).strip()
        md = md + "\n\n" + sm_local(md)

        out_path = os.path.join(SUB_MD_DIR, f"{sub}.md")
        safe_path = _safe_filename(out_path)
        # md = md.replace(f"""calls
        #            were""","calls were")
        # md = md.replace(f"""calls\nwere""","calls were")
        # md = md.replace(f"""calls\\nwere""","calls were")
        
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(md)
            
        save_md_pdf(md,sub+"8.pdf")
