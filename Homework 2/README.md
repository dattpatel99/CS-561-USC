# 5x5 Go Player

1. Problem:
-  Create a Go player agent that utilizes Alpha-Beta pruning to beat different kinds of Go players.
-  The broad size was 5x5
-  There was a maximum of 25 moves
-  Opponent Go agents include: Aggresive, Defensive, Minimax, Q-learning Agent, Mini-Alpha Go
2. Result:
- My agent was able to beat the Aggresive, Defensive and Minimax agents for all games. The Q-learning agent was beat for 80% of the games, and the Mini-Alpha Go agent was beat for 50%.
3. Method:
- Initially I hard coded some starting Go strategies to improve move selection, and then applied heuristics tah would alter the weight of decisions based on how many moves had been made.
- Furthermore, I applied a fixed depth of 3 for the alpha-beta prune to avoid running out of time.
4. Possible improvements:
- As we get closer to the finish we can track which moves lead to the most scenarios in which our agent wins and pick those.
