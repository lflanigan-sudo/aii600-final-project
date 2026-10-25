# Decisions

This file is the history of the project. Append an entry when a choice changes the work: the quantity, the data, the split, the model, the loss, or the action. Leave old entries in place. A later entry can reverse an earlier one. If a result depends on a random draw, record the seed under Why. The split entry records the cutoff date, or the seed and the fraction. The pull request that merges the code names this entry.

## 2026-10-23: Auction data and initial modeling scope

- Decision. Use ALT/FNData completed auction sales for Cartier, Rolex, and Omega at Sotheby’s and Bonhams, within the January 2021–October 2026 query window, to predict realized sale prices in USD.
- Why. The saved pull contains 5,000 rows representing 4,964 unique lots before eligibility cleaning. It provides observed sale outcomes and usable watch characteristics, although coverage varies and retrieval is capped and nonrandom.
- Alternative. Use asking-price listings or immediately include additional luxury brands. We selected completed sales to ground the target in observed transactions and limited the initial scope to establish a workable baseline.
- Revisit. Reconsider the scope if cleaning leaves insufficient coverage, source validation reveals unreliable prices, or held-out evaluation shows that the available characteristics cannot support useful predictions.
  
## 2026-10-23: Team Collaboration and GitHub Setup

- Decision. Use a shared GitHub repository with individual branches and pull requests for team contributions. Add all team members' names and GitHub usernames to README.md.

- Why. This follows Professor Sokolov's project requirements and allows team members to review changes before they are merged into the main branch.

- Alternative. Allow team members to edit the main branch directly. We rejected this because it would bypass the required peer-review process.

- Revisit. Update the team information if membership or GitHub usernames change. Continue using branches and pull requests throughout the project.

