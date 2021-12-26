using System;
//DO NOT DELETE the two following using statements *********************************
using Game_Logic_Class;
using Object_Classes;


namespace Space_Race
{

    class Console_Class
    {
        /// <summary>
        /// Algorithm below currently plays only one game
        /// 
        /// when have this working correctly, add the abilty for the user to 
        /// play more than 1 game if they choose to do so.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {      
             DisplayIntroductionMessage();
            /*                    
             Set up the board in Board class (Board.SetUpBoard)
             Determine number of players - initally play with 2 for testing purposes 
             Create the required players in Game Logic class
              and initialize players for start of a game             
             loop  until game is finished           
                call PlayGame in Game Logic class to play one round
                Output each player's details at end of round
             end loop
             Determine if anyone has won
             Output each player's details at end of the game
           */

            //The restart point in the case of user selecting yes for playing another game
            Restart:
            SpaceRaceGame.GameFinished = false;
            SpaceRaceGame.Players.Clear();

           //sets up board
            Board.SetUpBoard();

            //asks for the amount of player playing and passes the number to spaceracegame playercount
            askForPlayerAmount();

            //sets up players
            SpaceRaceGame.SetUpPlayers();

            //starts an infinite loop until a player wins or no players have fuel
            while (true)
            {
                displayPlayOneRoundOption();
                displayRoundName();
                SpaceRaceGame.PlayOneRound();
                //display all players location and fuel
                for(int tempVar = 0; tempVar < SpaceRaceGame.NumberOfPlayers; tempVar++)
                {
                    displayRoundResultText(tempVar); 
                }
                //breaks the loop when the game finishes
                if(SpaceRaceGame.GameFinished == true)
                {
                    break;
                }
            }

            //displays the player(s) who reached the final square
            playersWhoFinshedDisplay();

            //calls displayFinalStatistics
            displayFinalStatistics();

            //calls askForAnotherGame
            askForAnotherGame();
            goto Restart;
        }//end Main





//#######################################################################################
//---------------------------------------------------------------------------------------
        // DISPLAY FUNCTIONS  \/
//---------------------------------------------------------------------------------------
//#######################################################################################






        /// <summary>
        /// Display a welcome message to the console
        /// Pre:    none.
        /// Post:   A welcome message is displayed to the console.
        /// </summary>
        static void DisplayIntroductionMessage()
        {
            Console.WriteLine("\n\tWelcome to Space Race.\n");
        } //end DisplayIntroductionMessage

//-----------------------------------------------------------------------

        //displays text to play one round
        // Pre: none.
        // post: play round option displayed
        static void displayPlayOneRoundOption()
        {
            Console.WriteLine("\nPress Enter to play a round ...\n");
            Console.ReadKey();
        }

        //displays the round name
        //pre: nothing
        //post: either first or next round is displayed
        static void displayRoundName()
        {
            //diplays first round message then displays next round for the remainder
            if(SpaceRaceGame.Players[0].RocketFuel == Player.INITIAL_FUEL_AMOUNT)
            {
                Console.WriteLine("\tFirst Round\n");
            }
            else
            {
                Console.WriteLine("\tNext Round\n");
            }

        }

//-----------------------------------------------------------------------

        //displays players amount of fuel and location when called
        static void displayRoundResultText(int tempVar)
        {
            Console.WriteLine("\t{0} on square {1} with {2} yattowatt of power remaining", SpaceRaceGame.Players[tempVar].Name, SpaceRaceGame.Players[tempVar].Position, SpaceRaceGame.Players[tempVar].RocketFuel);
        }

//-----------------------------------------------------------------------

        //asks player for the amount of players they want in the game (must be bewteen 6 and 2)
        static void askForPlayerAmount()
        {
            //a point where 
            askForPlayerNumStart:
            Console.WriteLine("\tThis game is for 2 to 6 players.");
            Console.Write("\tHow many players (2-6): ");
            int playerCount;
            string userinput = Console.ReadLine();
            bool isCorrectInput = Int32.TryParse(userinput, out playerCount);
            // int playerCount = int.Parse(Console.ReadLine()); // add try PASS
            //checks if the user input is an int value (not null and is int)
            if (isCorrectInput == true && userinput != null)
            {
                // checks is user input is within the min and max amount of players
                if (playerCount > SpaceRaceGame.MAX_PLAYERS || playerCount < SpaceRaceGame.MIN_PLAYERS)
                {
                    while (playerCount > 6 || playerCount < 2)
                    {
                        Console.WriteLine("\tNumber of Players is not beetween 2-6\n");
                        Console.Write("\tHow many players (2-6): ");
                        playerCount = int.Parse(Console.ReadLine()); // add tryparse
                        Console.Write("\n\n");
                    }

                }
                SpaceRaceGame.NumberOfPlayers = playerCount;
            }
            else
            {
                //displays invalid input message
                Console.WriteLine("\n\tInvalid Input!\n");
                goto askForPlayerNumStart;
            }
        }


//-----------------------------------------------------------------------

        //displays all the players who reached the final square
        static void playersWhoFinshedDisplay()
        {
            Console.WriteLine("\n\n\tThe following player(s) finished the game\n");
            for (int tempVar = 0; tempVar < SpaceRaceGame.NumberOfPlayers; tempVar++)
            {
                if (SpaceRaceGame.Players[tempVar].AtFinish == true)
                {
                    Console.WriteLine("\t\t{0}\n", SpaceRaceGame.Players[tempVar].Name);
                }
            }
        }


//-----------------------------------------------------------------------


        //Displays the final location and fuel of all players
        static void displayFinalStatistics()
        {
            Console.WriteLine("Individual players finished with the at the locations specified");
            for (int tempVar = 0; tempVar < SpaceRaceGame.NumberOfPlayers; tempVar++)
            {
                Console.WriteLine("\n\t\t{0} with {1} yattowatt of power at square {2}\n", SpaceRaceGame.Players[tempVar].Name, SpaceRaceGame.Players[tempVar].RocketFuel, SpaceRaceGame.Players[tempVar].Position);
            }
        }


//-----------------------------------------------------------------------


        //when called displays message to terminate program, then waits for key press to terminate.
        static void PressEnter()
        {
            Console.WriteLine("\tPress Enter to termintate program ...");
            Console.ReadKey();
            Environment.Exit(-1);
        }


//-----------------------------------------------------------------------


        //displays promps to play another game
        //if "y" or "Y" is inputed, a new game starts
        //if anything else is inputed, the game pressEnter is called after message is displayed
        static void askForAnotherGame()
        {
            Console.WriteLine("\n\tPress Enter key to continue ...");
            Console.ReadKey();
            Console.Write("\n\n\n\tPlay Again? (Y or N): ");
            string playAnotherAnswer = Console.ReadLine();
            int IsYes;
            int IsLowerCaseY;
            IsYes = (String.Compare(playAnotherAnswer, "Y"));
            IsLowerCaseY = (String.Compare(playAnotherAnswer, "y"));
            if (IsYes == 0 || IsLowerCaseY == 0)
            {
                
            }
            else
            {
                Console.WriteLine("\n\n\n\tThanks for playing Space Race.\n");
                PressEnter();
            }
        } // end PressAny



    }//end Console class
}
