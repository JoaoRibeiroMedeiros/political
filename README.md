# Political Opinion Evolution Visualization

This project analyzes and visualizes the evolution of politicians' opinions over time regarding matters to be voted in congress.

## Visualizations

### All Politicians (Unordered)
![Unordered Opinion Evolution](modeldynamic.gif)
*Evolution of all politicians' opinions shown in an 18x18 grid*

### All Politicians (Sorted)
![Sorted Opinion Evolution](sortedmodeldynamic.gif)
*Evolution of all politicians' opinions, sorted by position*

### Party-Specific Evolution
![Party Opinion Evolution](party_modeldynamic.gif)
*Evolution of opinions within a specific party shown in a 5x5 grid*

### Color Legend
- 🔴 Red: Against (-1)
- ⚫ Gray: Silent/Neutral (0)
- 🟢 Green: In favor (1)

## Usage

To generate these visualizations, use the following functions:
- `animate_unordered()`: Creates unordered visualization of all politicians
- `animate_ordered()`: Creates sorted visualization of all politicians
- `animate_party()`: Creates party-specific visualization
