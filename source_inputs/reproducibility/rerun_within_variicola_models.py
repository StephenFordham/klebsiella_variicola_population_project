"""Within-K. variicola staged models, separation diagnostics and Firth inference.

Requires firthmodels==0.8.2 in st273_phylogeny_env. Its fitted model and profile
CI API are used directly. Block tests call its Newton solver with fixed terms,
retaining the full design's Jeffreys penalty, as required for nested Firth tests.
The block adapter is checked against the package's native one-parameter tests.
"""

from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import json
import warnings

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.optimize import linprog
from scipy.stats import chi2
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from firthmodels import FirthLogisticRegression
from firthmodels.logistic import compute_logistic_quantities, newton_raphson

from prepare_esbl_analysis_inputs import annotate_esbl, file_hash, parse_hits


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent.parent / "Klebsiella_variicola_curated_population.csv"
OUT = ROOT / "within_variicola_results"
OUTCOMES = ["ESBL", "MDR", "Carbapenemase"]
FORMULAS = {
    "Model 1": "C(source_category, Treatment(reference='environment'))",
    "Model 2": "C(source_category, Treatment(reference='environment')) + C(country_model) + year_centered",
    "Model 3": "C(source_category, Treatment(reference='environment')) + C(country_model) + year_centered + C(st_model)",
}


def prepare():
    d = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    assert d["Genome ID"].is_unique and d["Genome Name"].is_unique
    d = annotate_esbl(d)
    d["ESBL"] = d.ESBL_positive.astype(int)
    d["MDR"] = pd.to_numeric(d.resistance_class_count).ge(3).astype(int)
    d["Carbapenemase"] = d.Bla_Carb_acquired.map(lambda x: int(bool(parse_hits(x))))
    d["source_category"] = d["Source category"].str.strip().replace("", "unknown")
    d["year"] = pd.to_numeric(d["Collection year"], errors="coerce")
    complete = d.loc[d.year.notna()].copy()
    median = complete.year.median()
    complete["year_centered"] = complete.year - median
    mappings = []
    for original, target in [("Country", "country_model"), ("ST", "st_model")]:
        values = complete[original].str.strip().replace("", "Unknown")
        counts = values.value_counts()
        complete[target] = values.where(values.map(counts).ge(10), "Other")
        for label, n in counts.items():
            mappings.append(dict(variable=original, original_label=label, complete_case_n=int(n),
                                 model_label=label if n >= 10 else "Other"))
    pd.DataFrame(mappings).to_csv(OUT / "rare_category_mapping.csv", index=False)
    complete[["Genome ID", "Genome Name", "ST", "Country", "source_category", "year",
              "year_centered", "country_model", "st_model", *OUTCOMES]].to_csv(OUT / "modelled_genomes.csv", index=False)
    return d, complete, median


def separation_check(X, y):
    """Linear feasibility test for complete or quasi-complete separation."""
    signed = (2*np.asarray(y)-1)[:,None] * np.asarray(X)
    k = signed.shape[1]
    objective = np.r_[-signed.sum(axis=0), signed.sum(axis=0)]
    constraints = np.vstack([np.c_[-signed, signed], np.ones((1,2*k))])
    bound = np.r_[np.zeros(len(y)), 1.0]
    result = linprog(objective, A_ub=constraints, b_ub=bound, bounds=(0,None), method="highs")
    if not result.success:
        raise RuntimeError(f"Separation diagnostic failed: {result.message}")
    return bool(-result.fun > 1e-7), float(-result.fun)


def full_quantities(model, beta):
    X, y, weights, offset = model._fit_data
    return compute_logistic_quantities(X=X, y=y, beta=beta, sample_weight=weights,
                                      offset=offset, workspace=model._workspace,
                                      penalty_weight=model.penalty_weight)


def block_lrt(model, fixed):
    """Test beta[fixed]=0 under the same full-model penalty in both fits."""
    fixed = np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(len(model.coef_)), fixed)
    beta = np.zeros(len(model.coef_))
    def quantities(values):
        beta[free] = values
        q = full_quantities(model, beta)
        return SimpleNamespace(loglik=q.loglik, modified_score=q.modified_score[free],
                               fisher_info=q.fisher_info[np.ix_(free,free)])
    fit = newton_raphson(quantities, n_features=len(free), max_iter=1000,
                        max_step=2, max_halfstep=50, gtol=1e-6, xtol=1e-6,
                        beta_init=model.coef_[free].copy(), warn_on_failure=True)
    if not fit.converged or not np.isfinite(fit.loglik):
        raise RuntimeError("Constrained Firth model did not converge.")
    statistic = 2*(model.loglik_ - fit.loglik)
    if statistic < -1e-6:
        raise RuntimeError("Constrained likelihood exceeded full likelihood.")
    return dict(statistic=max(0.,statistic), df=len(fixed),
                p_value=chi2.sf(max(0.,statistic),len(fixed)),
                full_penalized_loglik=model.loglik_, constrained_penalized_loglik=fit.loglik,
                constrained_converged=True, constrained_iterations=fit.n_iter)


def firth_fit(X,y):
    fit = FirthLogisticRegression(fit_intercept=False, backend="numpy", max_iter=1000,
                                 max_step=2, max_halfstep=50, gtol=1e-6, xtol=1e-6).fit(X,y)
    score = full_quantities(fit,fit.coef_).modified_score
    assert fit.converged_ and np.isfinite(fit.coef_).all()
    assert np.max(np.abs(score)) < 1e-5
    return fit


def validate_adapter():
    X = pd.DataFrame({"Intercept":np.ones(6), "group":[0,0,0,1,1,1]})
    y = np.array([0,0,0,1,1,1])
    fit = firth_fit(X,y)
    np.testing.assert_allclose(np.exp(fit.coef_[1]),49.,rtol=1e-5)
    fit.lrt([1])
    check = block_lrt(fit,[1])
    np.testing.assert_allclose(check["p_value"],fit.lrt_pvalues_[1],rtol=1e-5)
    return dict(test="separated_2x2_analytic_OR_and_native_LRT",expected_OR=49.,
                fitted_OR=np.exp(fit.coef_[1]), passed=True)


def profile_effect(fit, X, index, endpoint, scope):
    intervals = fit.conf_int(method="pl",features=[index],max_iter=1000,tol=1e-6)
    fit.lrt([index])
    ci = np.exp(intervals[index])
    if not np.isfinite(ci).all() or not np.isfinite(fit.lrt_pvalues_[index]):
        raise RuntimeError(f"Profile inference failed: {endpoint}, {scope}, {X.columns[index]}")
    test = block_lrt(fit,[index])
    np.testing.assert_allclose(test["p_value"],fit.lrt_pvalues_[index],rtol=1e-4,atol=1e-10)
    return dict(endpoint=endpoint,scope=scope,model="Model 3",term=X.columns[index],
                coefficient=fit.coef_[index],odds_ratio=np.exp(fit.coef_[index]),
                ci_low=ci[0],ci_high=ci[1],p_value=fit.lrt_pvalues_[index],
                estimator="Firth",CI_method="profile penalized likelihood",native_LRT_validated=True)


def mle_records(X,y,endpoint,stage,scope):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = sm.GLM(y,X,family=sm.families.Binomial()).fit(maxiter=200)
    sep,score = separation_check(X,y)
    ci=m.conf_int()
    rows=[]
    for name,beta in m.params.items():
        with np.errstate(over="ignore"):
            odds=np.exp(beta)
            low,high=np.exp(ci.loc[name])
        rows.append(dict(endpoint=endpoint,scope=scope,model=stage,term=name,coefficient=beta,
                         odds_ratio=odds,ci_low=low,ci_high=high,p_value=m.pvalues[name],
                         estimator="ordinary_MLE",separation_detected=sep))
    info=dict(endpoint=endpoint,scope=scope,model=stage,n=len(y),events=int(sum(y)),
              parameters=X.shape[1],design_rank=int(np.linalg.matrix_rank(X)),
              reported_converged=bool(m.converged),separation_detected=sep,
              separation_objective=score,AUC=roc_auc_score(y,m.fittedvalues),
              loglik=m.llf,extreme_coefficients=int((m.params.abs()>15).sum()),
              warnings="; ".join(sorted({type(w.message).__name__ for w in caught})))
    return m,rows,info


def describe(data,complete):
    summaries, source_rows, category_rows = [],[],[]
    for endpoint in OUTCOMES:
        pos=data[data[endpoint].eq(1)]
        st=pos.ST
        known=~pos.Country.str.lower().isin(["","unknown","-","nan"])
        summary=dict(endpoint=endpoint,total_n=len(data),positive_n=len(pos),prevalence=len(pos)/len(data),
                     complete_case_n=len(complete),complete_case_positive_n=int(complete[endpoint].sum()),
                     positive_MLST_labels=st.nunique(),positive_exact_STs=st[st.str.fullmatch(r"ST\d+")].nunique(),
                     positive_variant_labels=st[st.str.fullmatch(r"ST\d+-\d+LV")].nunique(),
                     known_positive_countries=pos.loc[known,"Country"].nunique(),unknown_country_positive_n=int((~known).sum()),
                     ST347_positive_n=int(pos.ST.eq("ST347").sum()),
                     US_positive_n=int(pos.Country.eq("United States").sum()))
        summaries.append(summary)
        for scope,frame in [("population",data),("complete_cases",complete)]:
            for source,group in frame.groupby("source_category"):
                source_rows.append(dict(endpoint=endpoint,scope=scope,source=source,total_n=len(group),
                                        positive_n=int(group[endpoint].sum()),prevalence=group[endpoint].mean()))
        for variable in ["source_category","country_model","st_model"]:
            for level,group in complete.groupby(variable):
                events=int(group[endpoint].sum())
                category_rows.append(dict(endpoint=endpoint,variable=variable,level=level,
                                          n=len(group),events=events,nonevents=len(group)-events,
                                          homogeneous_outcome=events in [0,len(group)]))
    pd.DataFrame(summaries).to_csv(OUT/"outcome_population_summary.csv",index=False)
    pd.DataFrame(source_rows).to_csv(OUT/"source_prevalence.csv",index=False)
    pd.DataFrame(category_rows).to_csv(OUT/"sparse_outcome_category_diagnostics.csv",index=False)


def main():
    assert importlib.metadata.version("firthmodels")=="0.8.2"
    OUT.mkdir(exist_ok=True)
    sha=file_hash(INPUT)
    data,complete,year_median=prepare()
    describe(data,complete)
    validation=[validate_adapter()]
    metrics,coefficients,profile_rows,tests,mle_coefs,diagnostics,oldtests,posthoc=[],[],[],[],[],[],[],[]
    grouping=pd.read_csv(OUT/"rare_category_mapping.csv")
    print(f"Complete cases {len(complete)}, centred on year {year_median}",flush=True)
    for endpoint in OUTCOMES:
        models={}
        for stage,formula in FORMULAS.items():
            X=dmatrix(formula,complete,return_type="dataframe")
            y=complete[endpoint].to_numpy(dtype=int)
            assert np.linalg.matrix_rank(X)==X.shape[1]
            mle,rows,diag=mle_records(X,y,endpoint,stage,"full_population")
            mle_coefs.extend(rows); diagnostics.append(diag)
            fit=firth_fit(X,y)
            models[stage]=(fit,X,mle)
            metrics.append(dict(endpoint=endpoint,model=stage,n=len(y),events=int(y.sum()),parameters=X.shape[1],
                                AUC=roc_auc_score(y,fit.predict_proba(X)[:,1]),estimator="Firth",
                                AUC_type="apparent_in_sample",converged=fit.converged_,
                                iterations=fit.n_iter_,max_abs_modified_score=float(np.max(np.abs(full_quantities(fit,fit.coef_).modified_score)))))
            wald=fit.conf_int(method="wald")
            for j,name in enumerate(X.columns):
                coefficients.append(dict(endpoint=endpoint,model=stage,term=name,coefficient=fit.coef_[j],
                                         odds_ratio=np.exp(fit.coef_[j]),ci_low=np.exp(wald[j,0]),
                                         ci_high=np.exp(wald[j,1]),p_value=fit.pvalues_[j],
                                         inference="Wald diagnostics only; use profile_effects.csv for manuscript"))
            print(f"{endpoint} {stage}: Firth AUC {metrics[-1]['AUC']:.4f}; MLE separation={diag['separation_detected']}",flush=True)
        for fullstage,reducedstage,pattern in [("Model 2","Model 1",["country_model","year_centered"]),
                                               ("Model 3","Model 2",["st_model"])]:
            fit,X,mle=models[fullstage]
            fixed=[i for i,name in enumerate(X.columns) if any(token in name for token in pattern)]
            test=block_lrt(fit,fixed)
            test.update(endpoint=endpoint,comparison=f"{fullstage} vs {reducedstage}",test="penalized_LRT_full_design_penalty")
            tests.append(test)
            smaller=models[reducedstage][2]
            oldtests.append(dict(endpoint=endpoint,comparison=f"{fullstage} vs {reducedstage}",
                                 statistic=2*(mle.llf-smaller.llf),df=mle.df_model-smaller.df_model,
                                 p_value=chi2.sf(2*(mle.llf-smaller.llf),mle.df_model-smaller.df_model),
                                 status="MLE diagnostic only; separation present"))
        fit,X,mle=models["Model 3"]
        sourceidx=[i for i,name in enumerate(X.columns) if "source_category" in name]
        globaltest=block_lrt(fit,sourceidx)
        globaltest.update(endpoint=endpoint,comparison="Global source association in Model 3",test="penalized_LRT_full_design_penalty")
        tests.append(globaltest)
        yearidx=list(X.columns).index("year_centered")
        profile_rows.append(profile_effect(fit,X,yearidx,endpoint,"full_population_year"))
        # Post-hoc restriction preserves grouping from the main complete-case set.
        subset=complete[complete.source_category.isin(["human","environment"])].copy()
        Xsub=dmatrix(FORMULAS["Model 3"],subset,return_type="dataframe")
        ysub=subset[endpoint].to_numpy(dtype=int)
        assert np.linalg.matrix_rank(Xsub)==Xsub.shape[1]
        msub,rows,diag=mle_records(Xsub,ysub,endpoint,"Model 3","human_environment")
        mle_coefs.extend(rows); diagnostics.append(diag)
        subfit=firth_fit(Xsub,ysub)
        idx=next(i for i,c in enumerate(Xsub.columns) if "source_category" in c and "[T.human]" in c)
        row=profile_effect(subfit,Xsub,idx,endpoint,"human_environment")
        row["n_modelled"]=len(subset)
        for src in ["human","environment"]:
            group=subset[subset.source_category.eq(src)]
            row[f"{src}_n"]=len(group)
            row[f"{src}_positive_n"]=int(group[endpoint].sum())
        posthoc.append(row)
        print(f"{endpoint}: year OR {profile_rows[-1]['odds_ratio']:.3f}; human/environment OR {row['odds_ratio']:.3f}",flush=True)
        # Persist complete endpoints so lengthy runs remain reviewable.
        for filename,records in [("staged_model_metrics.csv",metrics),("firth_coefficients_Wald_diagnostics.csv",coefficients),
                                 ("profile_effects.csv",profile_rows),("penalized_likelihood_ratio_tests.csv",tests),
                                 ("MLE_coefficients_diagnostics.csv",mle_coefs),("model_stability_diagnostics.csv",diagnostics),
                                 ("MLE_likelihood_ratio_tests_diagnostics.csv",oldtests),
                                 ("human_environment_posthoc.csv",posthoc)]:
            pd.DataFrame(records).to_csv(OUT/filename,index=False)
    validation.extend(dict(test=f"native_LRT_vs_block_adapter_{r['endpoint']}_{r['scope']}",passed=r["native_LRT_validated"])
                      for r in profile_rows+posthoc)
    pd.DataFrame(validation).to_csv(OUT/"validation_checks.csv",index=False)
    assert file_hash(INPUT)==sha
    manifest=dict(input=str(INPUT),input_sha256=sha,n_population=len(data),n_modelled=len(complete),
                  median_collection_year=year_median,source_reference="environment",rare_level_threshold=10,
                  rare_grouping_population="complete-case population, before restricting post-hoc sources",
                  retained_country_labels=int(sum((grouping.variable=="Country")&(grouping.model_label!="Other"))),
                  retained_MLST_labels=int(sum((grouping.variable=="ST")&(grouping.model_label!="Other"))),
                  primary_estimator="Firth bias-reduced logistic regression",
                  ordinary_MLE="retained as diagnostics; not primary inference because separation detected",
                  tests="Penalized likelihood-ratio tests with constrained parameters and unchanged full-design Jeffreys penalty",
                  confidence_intervals="95% profile penalized likelihood for year and human/environment effects",
                  AUC="Apparent/in-sample discrimination; no external or cross-validation claimed",
                  multiple_testing="No new multiple-testing family introduced; predefined outcome/model comparisons reported as in the submitted design",
                  versions={p:importlib.metadata.version(p) for p in ["pandas","numpy","scipy","statsmodels","firthmodels","scikit-learn","matplotlib","patsy"]},
                  references=["https://doi.org/10.1093/biomet/80.1.27","https://doi.org/10.1002/sim.1047",
                              "https://github.com/jzluo/firthmodels","https://search.r-project.org/CRAN/refmans/logistf/html/anova.html"])
    (OUT/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Within-species analysis complete",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
