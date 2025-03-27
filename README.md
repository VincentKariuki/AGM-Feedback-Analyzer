# AGM Feedback Analyzer

A Python tool for automatically analyzing Annual General Meeting (AGM) feedback forms using NLP techniques. Processes raw feedback to generate sentiment analysis, topic modeling, and clustered insights.

## Features

- **Sentiment Analysis**: Measures positive/negative tone of feedback
- **Topic Modeling**: Identifies key discussion topics using LDA
- **Automatic Clustering**: Groups similar feedback together
- **Visualizations**: Generates word clouds and distribution charts
- **Excel Reports**: Creates organized output with multiple sheets
- **Automated report generation**

## How to Use (Directly on GitHub)

1. **Create a new repository** using this template
2. **Upload your data**:
   - Click "Add file" → "Upload files"
   - Upload your CSV file (ensure it has columns: `Went_Well`, `Did_Not_Go_Well`, `Improvement_Suggestions`, `Other_Comments`)
3. **Run the analysis**:
   - Go to GitHub Codespaces (click "Code" → "Open with Codespaces")
   - Or use GitHub Actions (set up workflow to run periodically)

## File Structure
--input sample_input.csv --output outputs/results.xlsx

## Requirements
The script requires these Python packages (automatically installed when using Codespaces):

pandas

scikit-learn

textblob

wordcloud

matplotlib

xlsxwriter

## Output Examples
Excel Report with sheets:

Raw Data

Topic Analysis

Sentiment Scores

Visualizations

Sample Visualization


## Contributing
Contributions welcome! Please:

Fork the repository

Create a new branch

Submit a pull request

## Optional
If you would like to save your analysis in an excel file you can add this piece of code at the end.



   if __name__ == "__main__":
       # Hardcode paths for Jupyter (or use interactive selection)
       input_path = r"C:\path\Cleaned_File.csv"  # ← Change this
       output_path = r"C:\path\File-Analyzed.xlsx"  # ← Change this
    
       # Run analysis without command-line arguments
       check_dependencies()
       analyze_feedback(
           input_path=input_path,
           output_path=output_path,
           num_topics=3,  # Customize as needed
           num_clusters=3  # Customize as needed
       )
