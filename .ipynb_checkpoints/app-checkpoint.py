import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 1. Native PyTorch XPU LLM Setup (Using Qwen 4)
# -------------------------------------------------------------------
@st.cache_resource
def load_llm():
    # Updated to Qwen 4 model
    model_id = "Qwen/Qwen3-1.7B" 
    
    # Check for native PyTorch XPU availability (No IPEX required as of PyTorch 2026 native support)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        st.sidebar.success(f"Hardware Acceleration: Native Intel XPU detected! ({torch.xpu.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        st.sidebar.warning("Intel XPU not found. Falling back to CPU.")

    # Load Qwen Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load Qwen Model natively to XPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    return tokenizer, model, device

# -------------------------------------------------------------------
# 2. Extract Data & Analyze Work Notes using Qwen 4
# -------------------------------------------------------------------
def analyze_fcr_with_llm(text, tokenizer, model, device):
    """
    Uses the local Qwen 4 LLM on Intel XPU to determine if the ticket was 
    resolved on the first follow-up based on work notes.
    """
    # Structuring the prompt using Qwen's chat template for highest accuracy
    messages =[
        {"role": "system", "content": "You are a precise IT support analyst. Your only job is to output 'YES' or 'NO'."},
        {"role": "user", "content": f"""Determine if the following customer ticket work notes indicate a "First Contact Resolution" (the issue was fixed in the very first reply to the customer without needing further back-and-forth).
        
Work Notes: {text}

Answer with ONLY "YES" if it was resolved on the first contact, or "NO" if it took multiple interactions."""}
    ]
    
    # Apply Qwen's specific conversational format
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            temperature=0.1, # Low temperature for strict classification
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
    
    return "YES" in response

# -------------------------------------------------------------------
# 3. Streamlit UI Dashboard
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Intel Customer Support KPI Dashboard", layout="wide")
    st.title("📊 Customer Support KPI: First Contact Resolution (FCR)")
    st.markdown("Target: **Close 70% of tickets in the first follow-up.** Powered by **Qwen Model** on Native PyTorch XPU.")

    # Load LLM
    with st.spinner("Loading Qwen model on Intel XPU..."):
        tokenizer, model, device = load_llm()

    # File uploader for the sample report
    uploaded_file = st.file_uploader("Upload Ticket Data (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        # Step 1: Extract Data
        df = pd.read_excel(uploaded_file)
        
        if "Work notes" not in df.columns or "Number" not in df.columns:
            st.error("Excel file must contain 'Number' and 'Work notes' columns.")
            return

        if st.button("Run FCR Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            fcr_results =[]
            
            # Step 2: Iterate and Analyze
            for idx, row in df.iterrows():
                status_text.text(f"Analyzing ticket {row['Number']} on XPU...")
                is_fcr = analyze_fcr_with_llm(row["Work notes"], tokenizer, model, device)
                fcr_results.append("Resolved on First Contact" if is_fcr else "Multiple Contacts Needed")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(df))
            
            df["FCR Status"] = fcr_results
            status_text.text("Analysis Complete!")
            
            # Calculate KPIs
            total_tickets = len(df)
            fcr_count = len(df[df["FCR Status"] == "Resolved on First Contact"])
            fcr_rate = (fcr_count / total_tickets) * 100 if total_tickets > 0 else 0

            # Step 3: Visualize Results
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("KPI Summary")
                st.metric(label="Total Tickets Analyzed", value=total_tickets)
                st.metric(label="First Contact Resolutions", value=fcr_count)
                
                # Highlight if target is met
                delta_color = "normal" if fcr_rate >= 70 else "inverse"
                st.metric(label="FCR Rate (Target: 70%)", value=fcr_rate, delta=fcr_rate - 70, delta_color=delta_color)

            with col2:
                # Gauge Chart for FCR Target
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = fcr_rate,
                    domain = {'x': [0, 1], 'y':[0, 1]},
                    title = {'text': "FCR Rate (%)", 'font': {'size': 24}},
                    delta = {'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range':[0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range':[0, 70], 'color': 'lightcoral'},
                            {'range': [70, 100], 'color': 'lightgreen'}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 70}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            # Display Data Table
            st.subheader("Ticket Details")
            st.dataframe(df, use_container_width=True)

            # Option to download the updated report
            output_xlsx = "fcr_analysis_report.xlsx"
            df.to_excel(output_xlsx, index=False)
            with open(output_xlsx, "rb") as file:
                st.download_button(
                    label="Download Automated Report",
                    data=file,
                    file_name=output_xlsx,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()