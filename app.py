import streamlit as st
import uuid
import fitz  # PyMuPDF
import cv2
import numpy as np
from rapidfuzz import fuzz
from google.cloud import vision
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Document Comparison Tool", layout="wide")

# --- Initialize Google Cloud Vision Client using Streamlit's secrets ---
# Streamlit automatically handles the secrets.toml file
try:
    credentials_dict = st.secrets["gcp_service_account"]
    vision_client = vision.ImageAnnotatorClient(credentials=credentials_dict)
except Exception as e:
    st.error(f"Error initializing Google Vision client. Ensure your secrets are set correctly. Details: {e}")
    st.stop()


# --- Core Logic Functions (from our Colab notebook) ---
SIMILARITY_THRESHOLD = 85

def file_to_image(uploaded_file, dpi=300):
    """Converts an uploaded file's content into an OpenCV image."""
    try:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=dpi)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            return cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Error converting {uploaded_file.name} to image: {e}")
        return None

def extract_text_blocks(image_content):
    """Uses the Google Vision API to get structured text blocks."""
    if not image_content: return []
    vision_image = vision.Image(content=image_content)
    response = vision_client.document_text_detection(image=vision_image)
    annotation = response.full_text_annotation
    if not annotation: return []

    blocks = []
    for page in annotation.pages:
        for block in page.blocks:
            block_text = "".join([symbol.text for paragraph in block.paragraphs for word in paragraph.words for symbol in word.symbols])
            block_text = " ".join(block_text.strip().split())
            if len(block_text) > 3:
                vertices = block.bounding_box.vertices
                blocks.append({
                    'id': f'block_{uuid.uuid4().hex[:6]}',
                    'text': block_text,
                    'bbox': (vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y)
                })
    return blocks

# --- Streamlit UI ---
st.title("📄 Document Comparison Tool")
st.write("Upload two documents (PDF or Image) to see a report of the text differences.")

col1, col2 = st.columns(2)

with col1:
    st.header("Original Document")
    file1 = st.file_uploader("Upload File 1", type=["pdf", "png", "jpg", "jpeg"], key="file1")

with col2:
    st.header("Revised Document")
    file2 = st.file_uploader("Upload File 2", type=["pdf", "png", "jpg", "jpeg"], key="file2")

if file1 and file2:
    if st.button("Compare Documents", use_container_width=True):
        with st.spinner("Processing documents... This may take a moment."):
            img1 = file_to_image(file1)
            img2 = file_to_image(file2)

            if img1 is not None and img2 is not None:
                st.info("Extracting text from both documents via Google Vision API...")
                blocks1 = extract_text_blocks(file1.getvalue())
                blocks2 = extract_text_blocks(file2.getvalue())

                st.info(f"Found {len(blocks1)} text blocks in File 1 and {len(blocks2)} text blocks in File 2.")
                
                # Perform stable two-way matching
                for b in blocks1: b['matched'] = False
                for b in blocks2: b['matched'] = False
                
                matches1_to_2 = {b1['id']: max(blocks2, key=lambda b2: fuzz.token_sort_ratio(b1['text'], b2['text']), default=None) for b1 in blocks1}
                matches2_to_1 = {b2['id']: max(blocks1, key=lambda b1: fuzz.token_sort_ratio(b2['text'], b1['text']), default=None) for b2 in blocks2}

                stable_matches = {}
                for id1, match2 in matches1_to_2.items():
                    if match2 is None: continue
                    id2 = match2['id']
                    if matches2_to_1.get(id2, {}).get('id') == id1:
                        score = fuzz.token_sort_ratio(next(b['text'] for b in blocks1 if b['id'] == id1), match2['text'])
                        if score >= SIMILARITY_THRESHOLD:
                            stable_matches[id1] = id2
                
                # Classify differences
                diffs = []
                for id1, id2 in stable_matches.items():
                    b1 = next((b for b in blocks1 if b['id'] == id1), None)
                    b2 = next((b for b in blocks2 if b['id'] == id2), None)
                    if b1 and b2:
                        b1['matched'], b2['matched'] = True, True
                        if b1['text'] != b2['text']:
                            diffs.append({'Type': 'Modified', 'Original Text': b1['text'], 'Revised Text': b2['text'], 'box1': b1['bbox'], 'box2': b2['bbox']})
                
                for b1 in blocks1:
                    if not b1['matched']:
                        diffs.append({'Type': 'Removed', 'Original Text': b1['text'], 'Revised Text': '---', 'box1': b1['bbox'], 'box2': None})
                for b2 in blocks2:
                    if not b2['matched']:
                        diffs.append({'Type': 'Added', 'Original Text': '---', 'Revised Text': b2['text'], 'box1': None, 'box2': b2['bbox']})

                st.success(f"Comparison complete. Found {len(diffs)} differences.")

                if diffs:
                    st.header("Difference Report")
                    df = pd.DataFrame(diffs)
                    st.dataframe(df[['Type', 'Original Text', 'Revised Text']], use_container_width=True)
                    
                    # Annotate images
                    img1_annotated = img1.copy()
                    img2_annotated = img2.copy()
                    colors = {"Modified": (0, 165, 255), "Removed": (0, 0, 255), "Added": (0, 200, 0)} # BGR

                    for d in diffs:
                        color = colors.get(d['Type'], (200, 200, 200))
                        if d.get('box1'): cv2.rectangle(img1_annotated, (d['box1'][0], d['box1'][1]), (d['box1'][2], d['box1'][3]), color, 3)
                        if d.get('box2'): cv2.rectangle(img2_annotated, (d['box2'][0], d['box2'][1]), (d['box2'][2], d['box2'][3]), color, 3)
                    
                    st.header("Annotated Images")
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.subheader("Original")
                        st.image(img1_annotated, channels="BGR")
                    with col_img2:
                        st.subheader("Revised")
                        st.image(img2_annotated, channels="BGR")
                else:
                    st.balloons()
                    st.success("No differences found!")
