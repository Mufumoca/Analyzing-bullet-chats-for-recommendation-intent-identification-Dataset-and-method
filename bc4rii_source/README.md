# Dataset and Method: Analyzing the Bullet Chats for Recommendation Intent Identification

## Part 1: Benchmark Dataset — **BC4RII**

**BC4RII** is the **first publicly available benchmark dataset** for *Recommendation Intent Identification (RII)* in livestream bullet chats.  
It contains **143,957 manually annotated bullet chats** collected from four major platforms: **Douyin, Kuaishou, Xiaohongshu, and TikTok**.  

The dataset is designed to help researchers distinguish between:
- **Purchase intent** (e.g., “这件有XL码吗？” / “Is there an XL size for this?”)
- **Casual chat** (e.g., “主播好美” / “The streamer looks beautiful”)

This work aims to support **real-time recommendation systems** by providing high-quality labeled data for intent recognition.

---

## Part 2: Method — **SPT-RII**

### Environment Setup
1. Install [OpenPrompt](https://github.com/thunlp/OpenPrompt).  
2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure that `fewshot.py`, `fewshot1.py`, `zeroshot.py`, and `autorun.py` have access to the **four required models**, downloaded and placed in the `model` directory.

---

### Usage Steps

1. **Explanation Generation**  
   Run `explanation_generation.py` to generate textual explanations for the bullet chats.

2. **Prepare the Dataset**  
   - The directory `BC4RII_with_generation` contains the dataset with generated explanations.  
   - For example, using the **RecDY** subset:  
     - Copy the training set  
       from:  
       ```
       BC4RII_with_generation/douyin/train.csv
       ```
       to:  
       ```
       datasets/TextClassification/rec-dy/train.csv
       ```
     - Copy the test set  
       from:  
       ```
       BC4RII_with_generation/douyin/test.csv
       ```
       to:  
       ```
       datasets/veb/rec-related/all_test.csv
       ```

3. **Run the Pipeline**  
   After completing the above steps, simply execute:
   ```bash
   python autorun.py
   ```

---

## Citation

If you find our dataset or method useful in your research, please cite our paper:

```
[coming soon...]
```
