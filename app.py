import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 1. Native PyTorch XPU LLM Setup
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
# 2. Extract Data & Analyze Work Notes using Strict Prompting
# -------------------------------------------------------------------
def analyze_fcr_with_llm(row, notes_col, created_col, closed_col, tokenizer, model, device):
    notes = str(row[notes_col]) if notes_col else "No notes provided"
    created = str(row[created_col]) if created_col and pd.notna(row[created_col]) else "Unknown"
    closed = str(row[closed_col]) if closed_col and pd.notna(row[closed_col]) else "Unknown"

    # Python calculates the exact duration to help the LLM understand the timeline
    duration_str = "Unknown"
    if created != "Unknown" and closed != "Unknown":
        try:
            created_dt = pd.to_datetime(created)
            closed_dt = pd.to_datetime(closed)
            minutes = (closed_dt - created_dt).total_seconds() / 60.0
            if minutes < 60:
                duration_str = f"{int(minutes)} minutes"
            else:
                duration_str = f"{round(minutes/60, 1)} hours"
        except Exception:
            pass

    # Highly structured prompt with strict constraints
    prompt = f"""You are a strict data classification AI. Your ONLY job is to output the exact word "YES" or "NO". Do not write anything else.

Read the following IT ticket data and determine if it is a First Contact Resolution (FCR).

FCR Rules:
- Output YES if the agent resolved the issue and closed the ticket without asking the customer for more information.
- Output YES if the "Time Elapsed" is very short (e.g., minutes). Multiple system logs from the SAME agent are fine.
- Output NO if the agent asked a question and had to wait for the customer to reply back.
- Output NO if the ticket was reassigned multiple times over several days.

Ticket Data:
- Time Elapsed: {duration_str}
- Work Notes: {notes[:1500]}

Is this an FCR? Answer ONLY YES or NO:"""

    messages =[
        {"role": "system", "content": "You are a strict data extraction assistant. You output only one word: YES or NO."},
        {"role": "user", "content": prompt}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=3,  # STRICLY limits output so it CANNOT ramble or "think"
            temperature=0.01,  # Near-zero temperature for absolute determinism
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
    
    # Simple, clean evaluation since we forced the model's hand
    is_fcr = response == "YES"

    return is_fcr, response

# -------------------------------------------------------------------
# 3. Streamlit UI Dashboard
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Intel Customer Support KPI Dashboard", layout="wide")
    st.title("📊 Customer Support KPI: First Contact Resolution (FCR)")
    st.markdown("Automated FCR calculation using Work Notes, Open, and Close dates. Powered by **Qwen3-1.7B** on Native PyTorch XPU.")

    with st.spinner("Loading Qwen3-1.7B on Intel XPU..."):
        tokenizer, model, device = load_llm()

    uploaded_file = st.file_uploader("Upload Ticket Data (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        # Auto-detect columns
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
                status_text.text(f"Analyzing ticket {ticket_id} on XPU... ({idx+1}/{len(df)})")
                
                is_fcr, raw_output = analyze_fcr_with_llm(
                    row, notes_col, created_col, closed_col, tokenizer, model, device
                )
                
                fcr_results.append("Resolved on First Contact" if is_fcr else "Multiple Contacts Needed")
                raw_llm_outputs.append(raw_output)
                
                progress_bar.progress((idx + 1) / len(df))
            
            df["FCR Status"] = fcr_results
            df["Raw LLM Output"] = raw_llm_outputs  
            status_text.text("Analysis Complete!")
            
            # Straightforward Percentage Calculation
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
                st.metric(label="FCR Percentage", value=f"{fcr_rate:.1f}%")

            with col2:
                # Clean Gauge Chart without the 70% delta logic
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fcr_rate,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Actual FCR Rate (%)", 'font': {'size': 24}},
                    number = {'suffix': "%", 'valueformat': ".1f"},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 100], 'color': 'lightgray'}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Ticket Details")
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