### Disability Data Cleaning Process - Publicly sourced Data

1. **Loading the Dataset**:  
   The dataset is loaded from a CSV file, containing information related to disability attributes like walking, hearing, and seeing difficulties.

2. **Filtering Based on Disability Indicator**:  
   The data is filtered to include only those rows where the `Disability_indicator` column has a value of `1`, representing individuals with disabilities.

3. **Filtering Based on Difficulty Levels**:  
   The dataset is further filtered to keep only those rows where there is a value of 2, 3, or 4 in any of the three difficulty attributes: `Difficulty_walking`, `Difficulty_hearing`, and `Difficulty_seeing`. This filtering helps focus on individuals with noticeable levels of difficulty in walking, hearing, or seeing.

4. **Selecting Relevant Columns**:  
   After filtering, only the essential columns are kept, which include `pid` (participant ID), `Disability_indicator`, and the three difficulty attributes (`Difficulty_walking`, `Difficulty_hearing`, `Difficulty_seeing`).

5. **Issue Classification**:  
   A new column, `issue`, is created to classify the type of difficulty based on the values in the difficulty attributes:
   - If the value is 3 or 4, the corresponding keyword (`mobility`, `dhh`, or `blv`) is appended.
   - If no attribute has a value of 3 or 4, but a value of 2 is present, the corresponding keyword for that attribute is included.
6. **Splitting the `issue` Column**:  
   The `issue` column is split into multiple columns (`issue_1`, `issue_2`, `issue_3`, etc.), where each issue type (e.g., mobility, dhh, blv) gets its own column.

7. **Saving the Processed Data**:  
   The cleaned and processed data is saved into a new CSV file for further analysis or use in other applications.

This process ensures that the dataset is focused on individuals with relevant disability difficulties, with clear classifications of the issues they face.
