Swap regret has been studied to death. Convexity is something that helps a lot, and our situation (finding the best temperature value) is convex, since there's some optimal in the middle and going away from that makes the predictions monotonically worse.

Ground truth: we want to sample from a temperature-adjusted *whole distribution* of pieces of text of length N. We could technically get this via beam search, followed by reweighing with the temperature.
 
Later, I realized that we are running into the label bias problem https://awni.github.io/label-bias/ and I tried to look into and see if addressing the temperature could partially address the label bias problem.
