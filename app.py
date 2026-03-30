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
    model_id = "Qwen/Qwen3-1.7B" 
    
    # Check for native PyTorch XPU availability (PyTorch 2026 native support)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        st.sidebar.success(f"Hardware Acceleration: Native Intel XPU detected! ({torch.xpu.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        st.sidebar.warning("Intel XPU not found. Falling back to CPU.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
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
def analyze_fcr_with_llm(row, notes_col, created_col, closed_col, tokenizer, model, device):
    """
    Evaluates FCR by looking at Work Notes alongside the Creation and Closure dates.
    Instructs the LLM that multiple system timestamps do not mean multiple customer interactions.
    """
    notes = str(row[notes_col]) if notes_col else "No notes provided"
    created = str(row[created_col]) if created_col and pd.notna(row[created_col]) else "Unknown"
    closed = str(row[closed_col]) if closed_col and pd.notna(row[closed_col]) else "Unknown"

    # Contextual Prompt instructing the LLM on ITSM realities
    prompt = f"""Analyze the following IT support ticket to determine if it meets First Contact Resolution (FCR).

FCR Definition: 
The issue was resolved by the agent without needing to request more information from the customer. 

CRITICAL RULES: 
1. "Work notes" often contain multiple system timestamps, assignment logs, or consecutive updates by the SAME agent. This does NOT mean multiple interactions! As long as the customer wasn't asked to reply back, it is an FCR.
2. Look at the Created and Closed times. If the ticket was resolved shortly after creation (e.g., within minutes or hours), it is highly likely an FCR.

Ticket Details:
- Created: {created}
- Closed: {closed}
- Work Notes:
{notes}

Answer with ONLY "YES" if it was resolved on the first contact, or "NO" if it required back-and-forth communication with the customer."""

    messages =[
        {"role": "system", "content": "You are an expert IT data analyst. Your only job is to evaluate tickets and output exactly 'YES' or 'NO'."},
        {"role": "user", "content": prompt}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            temperature=0.1, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
    
    # Return boolean status and the raw text for debugging
    return "YES" in response, response

# -------------------------------------------------------------------
# 3. Streamlit UI Dashboard
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Intel Customer Support KPI Dashboard", layout="wide")
    st.title("📊 Customer Support KPI: First Contact Resolution (FCR)")
    st.markdown("Target: **Close 70% of tickets in the first follow-up.** Powered by **Qwen 4** on Native PyTorch XPU.")

    # Load LLM
    with st.spinner("Loading Qwen 4 on Intel XPU..."):
        tokenizer, model, device = load_llm()

    uploaded_file = st.file_uploader("Upload Ticket Data (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        # Auto-detect columns based on the screenshot provided
        ticket_col = "Number" if "Number" in df.columns else "Ticket ID" if "Ticket ID" in df.columns else df.columns[0]
        notes_col = "Work notes" if "Work notes" in df.columns else "Work Notes" if "Work Notes" in df.columns else None
        created_col = "Created" if "Created" in df.columns else None
        closed_col = "Closed" if "Closed" in df.columns else None

        if not notes_col:
            st.error("Could not find a 'Work notes' column. Please check your Excel headers.")
            return

        if st.button("Run FCR Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            fcr_results =[]
            raw_llm_outputs = []
            
            for idx, row in df.iterrows():
                ticket_id = row[ticket_col]
                status_text.text(f"Analyzing ticket {ticket_id} on XPU...")
                
                is_fcr, raw_output = analyze_fcr_with_llm(
                    row, notes_col, created_col, closed_col, tokenizer, model, device
                )
                
                fcr_results.append("Resolved on First Contact" if is_fcr else "Multiple Contacts Needed")
                raw_llm_outputs.append(raw_output)
                
                progress_bar.progress((idx + 1) / len(df))
            
            df["FCR Status"] = fcr_results
            df["Raw LLM Output"] = raw_llm_outputs  # Added for transparency
            status_text.text("Analysis Complete!")
            
            # Calculate KPIs
            total_tickets = len(df)
            fcr_count = len(df[df["FCR Status"] == "Resolved on First Contact"])
            fcr_rate = (fcr_count / total_tickets) * 100 if total_tickets > 0 else 0

            # Visualize Results
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("KPI Summary")
                st.metric(label="Total Tickets Analyzed", value=total_tickets)
                st.metric(label="First Contact Resolutions", value=fcr_count)
                
                delta_color = "normal" if fcr_rate >= 70 else "inverse"
                st.metric(label="FCR Rate (Target: 70%)", value=fcr_rate, delta=fcr_rate - 70, delta_color=delta_color)

            with col2:
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

            st.subheader("Ticket Details")
            # Display dataframe highlighting the output
            st.dataframe(df[[ticket_col, created_col, closed_col, "FCR Status", "Raw LLM Output", notes_col]], use_container_width=True)

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