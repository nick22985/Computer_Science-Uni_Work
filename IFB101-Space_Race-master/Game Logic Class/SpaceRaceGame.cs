using System.Drawing;
using System.ComponentModel;
using Object_Classes;

namespace Game_Logic_Class
{
    public static class SpaceRaceGame
    {
        // Minimum and maximum number of players.
        public const int MIN_PLAYERS = 2;
        public const int MAX_PLAYERS = 6;
        private static int numberOfPlayers = 2;  //default value for test purposes only 
        public static int NumberOfPlayers
        {
            get
            {
                return numberOfPlayers;
            }
            set
            {
                numberOfPlayers = value;
            }
        }

        public static string[] names = { "One", "Two", "Three", "Four", "Five", "Six" };  // default values

        // Only used in Part B - GUI Implementation, the colours of each player's token
        private static Brush[] playerTokenColours = new Brush[MAX_PLAYERS] { Brushes.Yellow, Brushes.Red,
                                                                       Brushes.Orange, Brushes.White,
                                                                      Brushes.Green, Brushes.DarkViolet};
        /// <summary>
        /// A BindingList is like an array which grows as elements are added to it.
        /// </summary>
        private static BindingList<Player> players = new BindingList<Player>();
        public static BindingList<Player> Players
        {
            get
            {
                return players;
            }
        }

        // The pair of die
        private static Die die1 = new Die(), die2 = new Die();

//-----------------------------------------------------------------------

        /// <summary>
        /// Set up the conditions for this game as well as
        ///   creating the required number of players, adding each player 
        ///   to the Binding List and initialize the player's instance variables
        ///   except for playerTokenColour and playerTokenImage in Console implementation.
        ///   
        ///     
        /// Pre:  none
        /// Post:  required number of players have been initialsed for start of a game.
        /// </summary>
        public static void SetUpPlayers()
        {
            Players.Clear();
            //Creates speicfied players for game
            for (int i = 0; i < numberOfPlayers; i++)
            {
                players.Add(new Player(names[i]));
                players[i].RocketFuel = Player.INITIAL_FUEL_AMOUNT;
                players[i].Location = Board.Squares[0];
                players[i].PlayerTokenColour = playerTokenColours[i];
                players[i].HasPower = true;
            }
            // for number of players
            //      create a new player object
            //      initialize player's instance variables for start of a game
            //      add player to the binding list
        }

 //-----------------------------------------------------------------------

        //checks if game is finished
        //game is finished when player(s) are at the finish or all players are out of fuel
        private static void IsGameFinished()
        {
            int are_all_player_out_of_fuel = 0;

            foreach(Player player in Players)
            {
                if (player.HasPower == false)
                {
                    are_all_player_out_of_fuel += 1;
                }

                if (player.AtFinish || are_all_player_out_of_fuel == numberOfPlayers)
                {
                    GameFinished = true;
                }

            }
        }

//-----------------------------------------------------------------------

        //returns the players who finished the game
        //in the case of all players running out of fuel, message is returned stating no players finished
        //used in GUI
        public static string showgameresults()
        {
            bool did_player_finish = false;
            string finished_output = "";
            string finished_players = "";
            foreach (Player player in Players)
            {
                //determines the players who have finished and stores their name in a string
                if(player.AtFinish)
                {
                    finished_players += string.Format("\n\t\t{0}", player.Name);
                    did_player_finish = true;
                }
            }

            if (did_player_finish == true)
            {
                //returns the players who finished allong with a message
                finished_output += ("\n\n\tThe following player(s) finished the game\n");
                finished_output += (finished_players + "\n\n");
            }

            else
            {
                //returns no players finished
                finished_output += ("\n\n\tNo players finished the game\n");
            }
            return finished_output;
        }


 //-----------------------------------------------------------------------

        /// <summary>
        ///  Plays one round of a game
        ///  calls play function for each player
        ///  when round is finished, game checks if it is over or not
        /// </summary>
        public static void PlayOneRound()
        {
            //if single step selected in GUI, rounds are played on a player turn basis
            if (Step_Single == true)
            {
                if (players[PlayernumForSingleStep].HasPower == true)
                {
                    // Roll and move players
                    players[PlayernumForSingleStep].Play(die1, die2);

                    // check if it is that last player for the individual round
                    if (PlayernumForSingleStep == (NumberOfPlayers - 1))
                    {
                        PlayernumForSingleStep = 0;
                        IsGameFinished();
                    }
                    else
                    {
                        // this will move to the next player if the game is not over.
                        playernumforsinglestep++;
                    }
                }

            }
            //game plays on a round by round basis
            else 
            {
            for (int i = 0; i < numberOfPlayers; i++)
                {
                    if (players[PlayernumForSingleStep].HasPower == true)
                    {
                        players[i].Play(die1, die2);
                    }
                }
                IsGameFinished();
            }
        }//end SpaceRace


//-----------------------------------------------------------------------


        private static bool gamefinished;
        /// <summary>
        ///  Checks if game has ended
        /// </summary>
        public static bool GameFinished
        {
            get
            {
                return gamefinished;
            }
            set
            {
                gamefinished = value;
            }
        }


//-----------------------------------------------------------------------


        private static bool step_single = false;
        /// <summary>
        /// single step toggle
        /// </summary>
        /// 
        public static bool Step_Single
        {
            get
            {
                return step_single;
            }
            set
            {
                step_single = value;
            }
        }


//-----------------------------------------------------------------------

        private static int playernumforsinglestep = 0;
        /// <summary>
        /// current player for single step
        /// </summary>
        public static int PlayernumForSingleStep
        {
            get
            {
                return playernumforsinglestep;
            }
            set
            {
                playernumforsinglestep = value;
            }
        }
    }
}