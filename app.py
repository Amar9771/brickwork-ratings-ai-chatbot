import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Brickwork Free AI Chatbot", layout="centered")
st.title("🤖 Brickwork Ratings — Free AI Assistant")
st.markdown("Ask the assistant to generate a **Rating Rationale** based on company financials.")

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ----------------- Sidebar for Inputs -----------------
st.sidebar.header("📊 Company Financials")

company_name = st.sidebar.text_input("Company Name", "XYZ Ltd")
revenue = st.sidebar.number_input("Revenue (₹ Cr)", value=2200)
net_profit = st.sidebar.number_input("Net Profit (₹ Cr)", value=150)
ebitda = st.sidebar.number_input("EBITDA (₹ Cr)", value=300)
de_ratio = st.sidebar.number_input("Debt-Equity Ratio", value=0.8)
outlook = st.sidebar.selectbox("Outlook", ["Stable", "Positive", "Negative", "Developing"])
analyst = st.sidebar.text_input("Analyst Name", "Amar")
rating_date = st.sidebar.date_input("Rating Date")

# ----------------- Build Prompt -----------------
prompt = f"""
You are a senior credit analyst at Brickwork Ratings. Based on the following data, generate a detailed rating rationale:

- Company Name: {company_name}
- Revenue: ₹{revenue} Cr
- Net Profit: ₹{net_profit} Cr
- EBITDA: ₹{ebitda} Cr
- Debt-Equity Ratio: {de_ratio}
- Outlook: {outlook}

Include:
- Key financial strengths
- Risk factors
- Industry trends
- Rating rationale and outlook
- Analyst: {analyst}
- Rating Date: {rating_date.strftime('%d-%b-%Y')}
"""

answer = ""

# ----------------- Generate Button -----------------
if st.button("📝 Generate Rating Rationale"):
    with st.spinner("Analyzing company data and drafting rationale..."):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=1024, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display Result
    st.markdown("### 📄 Rating Rationale")
    st.success(f"Rating rationale generated for **{company_name}**.")
    st.write(answer)

    # ----------------- Generate PDF -----------------
    pdf_file = f"{company_name}_Rating_Rationale.pdf"

    def generate_pdf(text, filename):
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, height - 50, f"{company_name} – Rating Rationale")

        c.setFont("Helvetica", 11)
        text_obj = c.beginText(40, height - 80)
        text_obj.setLeading(14)

        for line in text.split('\n'):
            text_obj.textLine(line.strip())
        c.drawText(text_obj)
        c.showPage()
        c.save()

    generate_pdf(answer, pdf_file)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download as PDF",
            data=f,
            file_name=pdf_file,
            mime="application/pdf"
        )

    # Cleanup temporary file
    try:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
    except Exception as e:
        st.error(f"Error while deleting the PDF file: {str(e)}")
