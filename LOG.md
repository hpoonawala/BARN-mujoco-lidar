## 2/4/25
- NM with Clipping at 0.5 isn't as effective as 0.2, but decent
- Gradient descent with clipping at 0.2 and alpha=0.01 was working ok (see `simdata/steepest_descent_clip_small_alpha.png`)
- If we increase alpha to 0.1, SD fails `simdata/steepest_descent_clip_0pt1_alpha.png`)
- Testing NM with clipping 0.5 and small step: almost as good 
- Testing NM with clipping 0.5 and 0.1 step: quite good, not as good as 0.2

## 2/5/25
- The initialization of $A_{mat}$ with an added term $0.01\ I$ was introducing the bias drift to the right
