"""
app.py - Student Performance Prediction System (Streamlit)

Run:
    streamlit run app.py

The app loads models/best_model.joblib (created by `python -m src.train_model`)
and predicts a student's final Mathematics grade G3 (0-20) WITHOUT using the
earlier grades G1 and G2.
"""

import pandas as pd
import streamlit as st

from src.data_preprocessing import FIGURES_DIR, VALID_NUMERIC_RANGES
from src.predict import (
    METRICS_PATH,
    MODEL_PATH,
    ModelNotFoundError,
    grade_band,
    load_metrics,
    load_model,
    predict_grade,
)

st.set_page_config(page_title="Student Performance Prediction System", page_icon="🎓", layout="wide")

# Human-friendly labels for the coded values in the dataset
EDU = {0: "0 - None", 1: "1 - Primary (4th grade)", 2: "2 - 5th to 9th grade",
       3: "3 - Secondary", 4: "4 - Higher education"}
TRAVEL = {1: "1 - < 15 min", 2: "2 - 15-30 min", 3: "3 - 30 min-1 hour", 4: "4 - > 1 hour"}
STUDY = {1: "1 - < 2 hours", 2: "2 - 2-5 hours", 3: "3 - 5-10 hours", 4: "4 - > 10 hours"}
FAIL = {0: "0", 1: "1", 2: "2", 3: "3 or more"}
SCALE_5 = {1: "1 - Very low", 2: "2 - Low", 3: "3 - Medium", 4: "4 - High", 5: "5 - Very high"}
FAMREL = {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Excellent"}
HEALTH = {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Very good"}
JOBS = ["teacher", "health", "services", "at_home", "other"]
YES_NO = ["yes", "no"]


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics():
    return load_metrics()


def coded_select(label, mapping, key, default, help_text=None):
    """Selectbox that shows a friendly label but returns the numeric code."""
    options = list(mapping.keys())
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, format_func=lambda v: mapping[v],
                        key=key, help=help_text)


def choice(label, options, key, default, help_text=None):
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=key, help=help_text)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎓 Student Performance Prediction System")
st.write(
    "This app estimates a student's **final Mathematics grade (G3, scale 0-20)** from "
    "personal, family, school and lifestyle information, using a machine-learning model "
    "trained on the UCI *Student Performance* dataset (395 Portuguese secondary-school students). "
    "Earlier exam grades **G1 and G2 are deliberately not used**, so the model shows what can be "
    "estimated from background and study information alone."
)
st.info(
    "Academic demonstration only (CSE570). This is not a validated tool for making real "
    "decisions about students.", icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Load model + metrics (clear error if missing)
# ---------------------------------------------------------------------------
try:
    model = get_model()
except ModelNotFoundError:
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Train the model first by running:\n\n`python -m src.train_model`"
    )
    st.stop()
except Exception as exc:  # e.g. corrupted file / version mismatch
    st.error(f"The model file could not be loaded: {exc}")
    st.stop()

metrics = get_metrics()
if metrics is None:
    st.warning(f"Metrics file `{METRICS_PATH}` not found - model information will be limited.")
    metrics = {}
defaults = metrics.get("form_defaults", {})


def d(name, fallback):
    return defaults.get(name, fallback)


tab_predict, tab_model, tab_data = st.tabs(["🔮 Predict", "📊 Model performance", "📁 About the data"])

# ---------------------------------------------------------------------------
# Tab 1 - prediction form
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Student details")
    st.caption("Fields start at the most typical value in the training data. Change any of them.")

    with st.form("student_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Personal**")
            age_low, age_high = VALID_NUMERIC_RANGES["age"]
            age = st.number_input("Age", min_value=age_low, max_value=age_high,
                                  value=int(d("age", 16)), step=1)
            sex = choice("Gender", ["F", "M"], "sex", d("sex", "F"),
                         "F = female, M = male")
            address = choice("Home address type", ["U", "R"], "address", d("address", "U"),
                             "U = urban, R = rural")
            health = coded_select("Current health status", HEALTH, "health", int(d("health", 4)))
            romantic = choice("In a romantic relationship", YES_NO, "romantic", d("romantic", "no"))
            higher = choice("Wants to take higher education", YES_NO, "higher", d("higher", "yes"))

        with c2:
            st.markdown("**Family**")
            famsize = choice("Family size", ["LE3", "GT3"], "famsize", d("famsize", "GT3"),
                             "LE3 = 3 or fewer, GT3 = more than 3")
            Pstatus = choice("Parents' cohabitation status", ["T", "A"], "Pstatus",
                             d("Pstatus", "T"), "T = living together, A = apart")
            Medu = coded_select("Mother's education", EDU, "Medu", int(d("Medu", 3)))
            Fedu = coded_select("Father's education", EDU, "Fedu", int(d("Fedu", 2)))
            Mjob = choice("Mother's job", JOBS, "Mjob", d("Mjob", "other"))
            Fjob = choice("Father's job", JOBS, "Fjob", d("Fjob", "other"))
            guardian = choice("Guardian", ["mother", "father", "other"], "guardian",
                              d("guardian", "mother"))
            famrel = coded_select("Quality of family relationships", FAMREL, "famrel",
                                  int(d("famrel", 4)))
            famsup = choice("Family educational support", YES_NO, "famsup", d("famsup", "yes"))
            internet = choice("Internet access at home", YES_NO, "internet", d("internet", "yes"))

        with c3:
            st.markdown("**School & study**")
            school = choice("School", ["GP", "MS"], "school", d("school", "GP"),
                            "GP = Gabriel Pereira, MS = Mousinho da Silveira")
            reason = choice("Reason for choosing the school",
                            ["home", "reputation", "course", "other"], "reason", d("reason", "course"))
            traveltime = coded_select("Home-to-school travel time", TRAVEL, "traveltime",
                                      int(d("traveltime", 1)))
            studytime = coded_select("Weekly study time", STUDY, "studytime", int(d("studytime", 2)))
            failures = coded_select("Number of past class failures", FAIL, "failures",
                                    int(d("failures", 0)))
            abs_low, abs_high = VALID_NUMERIC_RANGES["absences"]
            absences = st.number_input("Number of school absences", min_value=abs_low,
                                       max_value=abs_high, value=int(d("absences", 4)), step=1)
            schoolsup = choice("Extra educational support from school", YES_NO, "schoolsup",
                               d("schoolsup", "no"))
            paid = choice("Extra paid Maths classes", YES_NO, "paid", d("paid", "no"))
            activities = choice("Extra-curricular activities", YES_NO, "activities",
                                d("activities", "yes"))
            nursery = choice("Attended nursery school", YES_NO, "nursery", d("nursery", "yes"))

        with st.expander("Lifestyle (free time, going out, alcohol)"):
            l1, l2 = st.columns(2)
            with l1:
                freetime = coded_select("Free time after school", SCALE_5, "freetime",
                                        int(d("freetime", 3)))
                goout = coded_select("Going out with friends", SCALE_5, "goout", int(d("goout", 3)))
            with l2:
                Dalc = coded_select("Workday alcohol consumption", SCALE_5, "Dalc", int(d("Dalc", 1)))
                Walc = coded_select("Weekend alcohol consumption", SCALE_5, "Walc", int(d("Walc", 2)))

        submitted = st.form_submit_button("Predict final grade", type="primary",
                                          use_container_width=True)

    if submitted:
        student = {
            "age": int(age), "Medu": Medu, "Fedu": Fedu, "traveltime": traveltime,
            "studytime": studytime, "failures": failures, "famrel": famrel,
            "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
            "health": health, "absences": int(absences),
            "school": school, "sex": sex, "address": address, "famsize": famsize,
            "Pstatus": Pstatus, "Mjob": Mjob, "Fjob": Fjob, "reason": reason,
            "guardian": guardian, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher,
            "internet": internet, "romantic": romantic,
        }
        try:
            result = predict_grade(model, student)
        except ValueError as exc:
            st.error(str(exc))
        else:
            grade = result["predicted_grade"]
            test_mae = metrics.get("final_test_metrics", {}).get("MAE")

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted final grade (G3)", f"{grade:.1f} / 20")
            r2.metric("Grade band", grade_band(grade))
            if test_mae is not None:
                r3.metric("Typical error (test MAE)", f"± {test_mae:.1f} points")

            if result["was_clipped"]:
                st.warning(
                    f"The model's raw output was {result['raw_prediction']:.2f}, which is outside "
                    "the valid 0-20 range, so it is shown limited to that range."
                )

            if grade < 10:
                st.error("The estimate is below the pass mark of 10. This student profile "
                         "resembles students who may benefit from extra academic support.")
            else:
                st.success("The estimate is at or above the pass mark of 10.")

            st.markdown("#### How to read this prediction")
            explanation = (
                f"- The model ({metrics.get('final_model', 'saved model')}) estimates the grade "
                "from patterns it learned in 316 training students.\n"
            )
            if test_mae is not None:
                explanation += (
                    f"- On students it had never seen (test set), its predictions were off by "
                    f"**{test_mae:.1f} grade points on average**, so treat the value as a rough "
                    f"range (about {max(0, grade - test_mae):.1f} to {min(20, grade + test_mae):.1f}), "
                    "not an exact score.\n"
                )
            perm = metrics.get("permutation_importance")
            if perm:
                top = [k for k, v in perm.items() if v > 0][:5]
                explanation += (
                    "- Inputs that influenced the model most on the test set: "
                    + ", ".join(f"`{t}`" for t in top) + ".\n"
                )
            explanation += ("- These are statistical associations in historical data, "
                            "not causes. Real outcomes depend on many factors not in the data.")
            st.markdown(explanation)

# ---------------------------------------------------------------------------
# Tab 2 - model information and metrics
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("About the model")
    if metrics:
        ftm = metrics["final_test_metrics"]
        btm = metrics["baseline_test_metrics"]
        st.markdown(
            f"- **Final model:** {metrics['final_model']}\n"
            f"- **Best hyperparameters (GridSearchCV, 5-fold CV):** `{metrics['best_params']}`\n"
            f"- **Training / test students:** {metrics['n_train']} / {metrics['n_test']} "
            f"(random seed {metrics['random_state']})\n"
            f"- **Inputs:** {len(metrics['features'])} features (G1 and G2 excluded)\n"
            f"- **Pipeline:** StandardScaler (numbers) + OneHotEncoder (categories) + model"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test MAE", f"{ftm['MAE']:.2f}", f"{ftm['MAE'] - btm['MAE']:+.2f} vs baseline",
                  delta_color="inverse")
        m2.metric("Test RMSE", f"{ftm['RMSE']:.2f}", f"{ftm['RMSE'] - btm['RMSE']:+.2f} vs baseline",
                  delta_color="inverse")
        m3.metric("Test MSE", f"{ftm['MSE']:.2f}")
        m4.metric("Test R²", f"{ftm['R2']:.3f}", f"{ftm['R2'] - btm['R2']:+.3f} vs baseline")

        st.markdown("#### Model comparison")
        table = pd.DataFrame(metrics["comparison"]).T[["MAE", "RMSE", "R2", "CV_MAE", "CV_RMSE", "CV_R2"]]
        table.columns = ["Test MAE", "Test RMSE", "Test R²", "CV MAE", "CV RMSE", "CV R²"]
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
        st.caption("CV = mean of 5-fold cross-validation on the training set. Lower MAE/RMSE "
                   "and higher R² are better. The baseline always predicts the training mean.")
    else:
        st.write("No metrics available.")

    figs = [("13_model_comparison.png", "Model comparison"),
            ("14_actual_vs_predicted.png", "Actual vs predicted"),
            ("15_residuals.png", "Residuals"),
            ("16_permutation_importance.png", "Feature importance")]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figs):
        path = FIGURES_DIR / fname
        if path.exists():
            cols[i % 2].image(str(path), caption=caption, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3 - dataset information
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset")
    st.markdown(
        "- **Source:** UCI Machine Learning Repository - *Student Performance* "
        "([link](https://archive.ics.uci.edu/dataset/320/student+performance))\n"
        "- **Citation:** Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary "
        "School Student Performance. *Proc. 5th FUture BUsiness TEChnology Conference*, 5-12.\n"
        "- **File used:** `student-mat.csv` (Mathematics) - 395 students, 33 columns\n"
        "- **Target:** `G3` final grade, 0-20 (10 or more is a pass)\n"
        "- **Why G1/G2 are excluded:** they are earlier grades of the same course and are almost "
        "the same as G3, which would make the task artificially easy and less useful early in a "
        "school year."
    )
    for fname, caption in [("01_g3_distribution.png", "Distribution of G3"),
                           ("09_failures_vs_g3.png", "Past failures vs G3")]:
        path = FIGURES_DIR / fname
        if path.exists():
            st.image(str(path), caption=caption, width=650)
