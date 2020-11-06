# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from math import inf
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        if len(legalMoves) > 1 and 'Stop' in legalMoves:
            legalMoves.remove('Stop')
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(legalMoves[chosenIndex], scores[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        nearestFood = float("inf")
        for x in range(0, newFood.width, 1):
            for y in range(0, newFood.height, 1):
                if newFood[x][y]:
                    distance = util.manhattanDistance(newPos, (x, y))
                    if distance < nearestFood:
                        nearestFood = distance
        nearestFood = 1/nearestFood if nearestFood != 0 else float("inf")

        if currentGameState.getScore() < successorGameState.getScore():
            scoreIncrease = float("inf")
        else:
            scoreIncrease = 0

        ghostIsNear = 0
        for ghostPos in successorGameState.getGhostPositions():
            if util.manhattanDistance(newPos, ghostPos) == 0:
                ghostIsNear = -float('inf')

        score = ghostIsNear + scoreIncrease + nearestFood
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.actionToReturn = None

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def _value(gameState, agentIndex=0, currentDepth=0):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                if currentDepth == self.depth:
                    return self.evaluationFunction(gameState)
                return _maxValue(gameState, agentIndex, currentDepth + 1)
            else:
                return _minValue(gameState, agentIndex, currentDepth)

        def _maxValue(gameState, agentIndex=0, currentDepth=0):
            bestValue = -float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = _value(successor, (agentIndex + 1) % gameState.getNumAgents(), currentDepth)
                bestValue = max(bestValue, value)
                if value == bestValue and currentDepth == 1:
                    self.actionToReturn = action
            return bestValue

        def _minValue(gameState, agentIndex=0, currentDepth=0):
            bestValue = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = _value(successor, (agentIndex + 1) % gameState.getNumAgents(), currentDepth)
                bestValue = min(bestValue, value)
            return bestValue

        _value(gameState)
        return self.actionToReturn

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def _value(gameState, alpha=-float("inf"), beta=float("inf"), agentIndex=0, currentDepth=0):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                if currentDepth == self.depth:
                    return self.evaluationFunction(gameState)
                return _maxValue(gameState, alpha, beta, agentIndex, currentDepth + 1)
            else:
                return _minValue(gameState, alpha, beta, agentIndex, currentDepth)

        def _maxValue(gameState, alpha, beta, agentIndex=0, currentDepth=0):
            bestValue = -float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = _value(successor, alpha, beta, (agentIndex + 1) % gameState.getNumAgents(), currentDepth)
                bestValue = max(bestValue, value)
                if value == bestValue and currentDepth == 1:
                    self.actionToReturn = action
                if bestValue > beta:
                    return bestValue
                alpha = max(alpha, bestValue)
            return bestValue

        def _minValue(gameState, alpha, beta, agentIndex=0, currentDepth=0):
            bestValue = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = _value(successor, alpha, beta, (agentIndex + 1) % gameState.getNumAgents(), currentDepth)
                bestValue = min(bestValue, value)
                if bestValue < alpha:
                    return bestValue
                beta = min(beta, bestValue)
            return bestValue

        _value(gameState)
        return self.actionToReturn

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def _value(gameState, agentIndex=0, currentDepth=0):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                if currentDepth == self.depth:
                    return self.evaluationFunction(gameState)
                return _maxValue(gameState, agentIndex, currentDepth + 1)
            else:
                return _expValue(gameState, agentIndex, currentDepth)

        def _maxValue(gameState, agentIndex=0, currentDepth=0):
            bestValue = -float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = _value(successor, (agentIndex + 1) % gameState.getNumAgents(), currentDepth)
                bestValue = max(bestValue, value)
                if value == bestValue and currentDepth == 1:
                    self.actionToReturn = action
            return bestValue

        def _expValue(gameState, agentIndex=0, currentDepth=0):
            totalValue = 0
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                totalValue += (1 / len(actions)) * _value(successor, (agentIndex + 1) % gameState.getNumAgents(),
                                                              currentDepth)
            return totalValue

        _value(gameState)
        return self.actionToReturn

def betterEvaluationFunction(s):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def _closestItem(pacmanPos, itemPositions):
        closestDistance = float("inf")
        for itemPosition in itemPositions:
            distance = util.manhattanDistance(pacmanPos, itemPosition)
            if distance < closestDistance:
                closestDistance = distance
        return closestDistance

    food = s.getFood().asList()
    capsules = s.getCapsules()
    ghosts = s.getGhostStates()
    closestFood = _closestItem(s.getPacmanPosition(), food)
    closestCapsule = _closestItem(s.getPacmanPosition(), capsules)

    frenlyGhosts = list()
    dangerousGhosts = list()
    for ghost in ghosts:
        distance = util.manhattanDistance(s.getPacmanPosition(), ghost.getPosition())
        if 0 < ghost.scaredTimer:
            if ghost.scaredTimer > distance:
                frenlyGhosts.append(distance)
        else:
            dangerousGhosts.append(distance)
    closestDangerousGhost = min(dangerousGhosts) if len(dangerousGhosts) > 0 else float("inf")
    closestFrenlyGhost = min(frenlyGhosts) if len(frenlyGhosts) > 0 else float("inf")

    def _foodNear():
        return 1/closestFood

    def _capsuleNear():
        return 1/closestCapsule

    def _capsuleEaten():
        return 1/len(capsules) if len(capsules) > 0 else 1e5

    def _ghostNear():
        if closestDangerousGhost == 0:
            return -float("inf")
        elif len(frenlyGhosts) > 1:
            return 1/closestFrenlyGhost
        else:
            return 0

    score = _foodNear() + _capsuleNear() + 2*_ghostNear() + 100*_capsuleEaten() + s.getScore()
    return score

# Abbreviation
better = betterEvaluationFunction
