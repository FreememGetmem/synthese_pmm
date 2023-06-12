from streamlit_echarts import st_echarts
import streamlit as st
import pandas as pd
import pickle


rl = pickle.load(open('models/ExtraTreesRegressor_model_bon.pkl', 'rb'))

# ---- General things Productions Maraîchères Mailhot
st.title('PMM Dashboard')
submit = None
submit2 = None

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv()

# ---- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Model analysis"])
st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Adja & Mor]")
# st.sidebar.image("/imgs/logo.png")

if page == "Predictor":
    # --- Inputs
    st.markdown("Select input file to predict.")
    upload_columns = st.columns([3, 1])
    # File upload
    file_upload = upload_columns[0].expander(label="Upload a file")
    uploaded_file = file_upload.file_uploader("Choose a file", type=['xlsx'])

    if uploaded_file:
        # Save it as temp file
        temp_filename = "temp.xlsx"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_excel(temp_filename)
        st.write(df)

    if uploaded_file:
        st.info("The dataset to predict is loaded ! ")
        submit = upload_columns[1].button("Get predictions")

    # --- Submission
    st.markdown("""---""")
    if submit:
        with st.spinner(text="Fetching model prediction..."):
            predictions = rl.predict(df)
        # --- Ouputs
        outputs = st.columns([2, 1])
        outputs[0].markdown("Poids Net Prediction: ")
        df['PoisNet Predits'] = predictions
        st.write(df)
        csv = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv',
        )
        # submit2 = outputs[0].button("Get Files")
        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": [0.7, 0.5, 0.4, 0.3]}]
        }
        st_echarts(liquidfill_option)
else:
    import streamlit as st
    import streamlit.components.v1 as components

    st.header("test html import")

    HtmlFile = open("../visualization/report.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code)