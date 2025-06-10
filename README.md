# SDT and Delta Plot Analysis - Final Assignment

### Main Analysis
- `sdt_analysis.py` - Complete analysis code implementing SDT model and delta plots

### Results
- `results/1_posterior_analysis.png` - Analysis which displays the main findings
- `results/2_delta_plots_participant_1.png` - Response time analysis
- `results/3_sdt_results.txt` - Summary of key findings and statistics in text

## Overall Conclusiom
 Making the task harder had a large effect on how well people could detect signals. Making the stimuli more complex had almost no effect.

 ## Explanation using the data: 

- **Difficulty Effect**: 2.944 points
- **Stimulus Type Effect**: 0.030 points

 **Data**

- 79,903 total trials across 10 people
- 81.6% accuracy
- 4 conditions: Easy Simple, Easy Complex, Hard Simple, Hard Complex

**SDT Results**

**Easy Conditions**:
- Easy Simple: 4.73 points
- Easy Complex: 4.70 points
    - Average: 4.72 points

**Hard Conditions**:
- Hard Simple: 1.78 points  
- Hard Complex: 1.76 points
    - Average: 1.77 points

#### Interpretation
- Higher numbers = better at detecting signals
- Easy conditions were about 3x better than hard conditions
- Simple vs Complex made almost no difference (4.73 vs 4.70 in easy, 1.78 vs 1.76 in hard)

###  Posterior Analysis

#### Top Left Graph: How Good People Were at Each Condition
- 2 groups: Easy conditions (right side, high scores) and Hard conditions (left side, low scores)
- Blue and Orange (Easy conditions) are almost identical, showing stimulus type doesn't matter much
- Green and Red (Hard conditions) are also almost identical, meaning stimulus has almost no effect
- No overlap between easy and hard groups, showing the difficulty effect is real and strong

#### Top Right Graph:
- **Easy conditions**: For easy conditions, people were more careful or picky about their choices (higher bars on right)
- **Hard conditions**: For hard conditions, people were less careful or picky about their choices (lower bars on left)

#### Bottom Left Graph: Individual Differences
- Each bar = one person
- Shows some people are naturally better at this task than others
- Range: From about 2.0 to 5.0 points
- **Important**: Even though there are differences among people, they all show the same pattern (difficulty matters, stimulus type doesn't)

#### Bottom Right Graph: 
- **Red bar (Difficulty)**: Huge effect (3.1 points)
- **Blue bar (Stimulus Type)**: Tiny effect (0.03 points)
- **Conclusion**: Difficulty effect is about 100 times bigger than stimulus type effect

### Response Time Patterns (Delta Plots):

This figure shows how long it took people to respond, which tells us about their thinking process.

#### Upper Row (Overall Response Times)
**When comparing Hard vs Easy conditions**:
- **Lines go UP steeply** = Hard conditions take much longer, especially for slow responses
- Shows difficulty doesn't just slow people down a little bit but it entirely changes how they think

**When comparing Complex vs Simple conditions**:
- Lines are mostly flat, meaning complex stimuli added a small, constant delay
- Shows complexity just adds a bit of extra processing time, doesn't change the core decision

#### Lower Row (Correct vs Error Responses)
- Green lines (Correct) and Red lines (Errors) often overlap
- Shows that difficulty affects both right and wrong answers similarly
- What this means : Difficulty affects the whole decision process, not just accuracy

### Conclusion & Interpretations 

#### Easy vs Hard (The Big Effect)
When tasks are hard people are worse at detecting signals, their response times also change dramatically, and correct/incorrect responses are affected. Ultimately, based on the data, difficulty affects mental processes.

#### Simple vs Complex (The Small Effect)  
When stimuli are complex, people's detection ability is not really affected, their respnse times increase slightly but constantly and finally, there is a pattern for all response types. Therefore, complexity only affects surface-level processing.

### Accuracy

**Statistical Reliability**: Excellent
- **Convergence**: Perfect (R-hat = 1.007, should be < 1.1) 
- **Sample size**: Large enough (ESS = 1,901) 
- **Consistency**: Both analysis methods agree

**The results are statistically solid and trustworthy**

### Key Findings
- **Difficulty effect**: 2.944 d' units (large effect on discrimination ability)
- **Stimulus type effect**: 0.030 d' units (minimal effect on discrimination ability)  
- Both SDT and delta plot analyses converge on the conclusion that difficulty dominates in comparison to stimulus type effects

### Model Convergence
- **R-hat**: 1.007 (excellent convergence)
- **ESS**: 1,901 (effective sample size)

Model converged perfectly, therefore, results are statistically reliable.