/**
 * @file      farm.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 * 
 * @author    David Bayer \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            ibayer@fit.vutbr.cz
 *
 * @brief     PC lab 8 / MPI Farm - Sieve of Eratosthenes
 *
 * @version   2024
 *
 * @date      12 February  2020, 17:21 (created) \n
 * @date      18 April     2023, 19:53 (revised) \n
 * @date      16 April     2024, 17:00 (revised) \n
 *
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <sstream>
#include <thread>
#include <vector>

#include <mpi.h>
#include <scorep/SCOREP_User.h>

/// constant used for mpi printf
constexpr int MPI_ALL_RANKS = -1;
/// constant for root rank
constexpr int MPI_ROOT_RANK = 0;

//--------------------------------------------------------------------------------------------------------------------//
//                                            Helper function prototypes                                              //
//--------------------------------------------------------------------------------------------------------------------//
/// Wrapper to the C printf routine and specifies who can print (a given rank or all).
template<typename... Args>
void mpiPrintf(int who, const std::string_view format, Args&&... args);
/// Flush standard output.
void mpiFlush();

/// Return MPI rank in a given communicator.
int mpiGetCommRank(MPI_Comm comm);
/// Return size of the MPI communicator.
int mpiGetCommSize(MPI_Comm comm);


//--------------------------------------------------------------------------------------------------------------------//
//                                                 Helper functions                                                   //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * C printf routine with selection which rank prints.
 * @tparam Args  - variable number of arguments.
 * @param who    - which rank should print. If -1 then all prints.
 * @param format - format string.
 * @param ...    - other parameters.
 */
template<typename... Args>
void mpiPrintf(int who, const std::string_view format, Args&&... args)
{
  if ((who == MPI_ALL_RANKS) || (who == mpiGetCommRank(MPI_COMM_WORLD)))
  {
    if constexpr (sizeof...(args) == 0)
    {
      std::printf("%s", std::data(format));
    }
    else
    {
      std::printf(std::data(format), std::forward<Args>(args)...);
    }
  }
}// end of mpiPrintf
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stdout and call barrier to prevent message mixture.
 */
void mpiFlush()
{
  std::fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  // A known hack to correctly order writes
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

}// end of mpiFlush
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int mpiGetCommRank(MPI_Comm comm)
{
  int rank{};

  MPI_Comm_rank(comm, &rank);

  return rank;
}// end of mpiGetCommRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int mpiGetCommSize(MPI_Comm comm)
{
  int size{};

  MPI_Comm_size(comm, &size);

  return size;
}// end of mpiGetCommSize
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                          Sieve the interval sequentially                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Mutual message exchange between two partners using blocking exchange <start, stop).
 * @param [in]  start  - Minimum tested number.
 * @param [in]  stop   - Maximum tested number.
 * @param [out] primes - Vector of primes.
 */
void sieve(const int         start,
           const int         stop,
           std::vector<int>& primes)
{
  // Vector for results
  primes.clear();
  primes.reserve(stop - start);

  // Test all numbers
  for (int number = start; number < stop; number++)
  {
    bool       isPrime    = true;
    const int  maxDivisor = static_cast<int>(std::sqrt(number)) + 1;

    // Test one prime
    for (int divisor = 2; divisor < maxDivisor; divisor++)
    {
      if (number % divisor == 0)
      {
        isPrime = false;
        break;
      }
    }

    if (isPrime)
    {
      primes.push_back(number);
    }
  }// for number

}// end of sieve
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                             Parallel version using Farm                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Different tags for MPI messages
 */
enum class mpiTag : int
{
  /// Result data (vector of primes).
  result    = 0,
  /// Worker's request for new job.
  askForJob = 1,
  /// Farmer's response with new job.
  jobData   = 2,
  /// Nothing to process, stop message.
  stop      = 3
};// end of mpiTag
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Implement following functions                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * Farmer process for the sieve.
 *
 * 1. Run until having some data and not received all results.
 * 2. Wait for a message from a worker.
 * 3. Check the message.
 * 4. If it is a request for new job, generate one, send it to the worker and increment the number of message in the fly
 * 5. If there is no other jobs, send a stop message.
 * 6. If you receive a job result, store data into a file and increment the number of primes.
 *
 * @param [in] start   - Minimum tested number.
 * @param [in] stop    - Maximum tested number.
 * @param [in] jobSize - Job size.
 * @return number of primes.
 */
int farmer(const int start,
           const int stop,
           const int jobSize)
{
  // Where to write primes
  MPI_File         outputFile{MPI_FILE_NULL};
  std::string_view outFilename{"primes.txt"};

  // Open file from the farmer.
  MPI_File_delete(outFilename.data(), MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_SELF, outFilename.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &outputFile);

  // Number of primes found.
  int nPrimes{};
  // Actual position in the sequence.
  int actPosition = start;

  // Vector with primes received from the workers.
  std::vector<int> primes(jobSize);

  // Which workers are processing a job [true = has a job, false = idle], include farmer but don't count it.
  bool workerStatus[mpiGetCommSize(MPI_COMM_WORLD)];
  for (int i = 0; i < workerStatus[i]; i++)
  {
    workerStatus[i] = false;
  }

  // Lambda counting the number of active workers
  auto nActiveWorkers = [](const bool* workerStatus) -> int
  {
    int nActive{};
    // Skip farmer
    for (int rank = 1; rank < mpiGetCommSize(MPI_COMM_WORLD); rank++)
    {
      nActive += int(workerStatus[rank]);
    }
    return nActive;
  };


  //  There are jobs unprocessed OR there are workers still working on jobs.
  while ((actPosition < stop) || (nActiveWorkers(workerStatus) > 0))
  {
    MPI_Status probeStatus{};

    // Wait for any incoming message.
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &probeStatus);

    // Store rank of receiver into slave_rank
    const int workerRank = probeStatus.MPI_SOURCE;

    // Decide according to the tag which type of message we have got
    if (mpiTag(probeStatus.MPI_TAG) == mpiTag::askForJob)
    {
      int        jobRequestMessage{};
      MPI_Status jobRequestStatus{};

      MPI_Recv(&jobRequestMessage, 1, MPI_INT, workerRank, int(mpiTag::askForJob), MPI_COMM_WORLD, &jobRequestStatus);

      // I still have a job to be processed.
      if (actPosition < stop)
      {
        int job[2];   // start, stop values
        // Set job
        job[0] = actPosition;
        job[1] = ((actPosition + jobSize) < stop) ? actPosition + jobSize : stop;

        // Send job to a worker
        MPI_Send(job, 2, MPI_INT, workerRank, int(mpiTag::jobData), MPI_COMM_WORLD);
        // Mark that worker with as busy.
        workerStatus[workerRank] = true;
        actPosition += jobSize;
      }
      else
      {
        const int stopMessage = 1;
        // Send stop message to slave
        MPI_Send(&stopMessage, 1, MPI_INT, workerRank, int(mpiTag::stop), MPI_COMM_WORLD);

        // Mark a worker with with rank slave_rank as stopped */
        workerStatus[workerRank] = false;
      }
    }
    else
    {
      int        recvPrimes{};
      MPI_Status resultStatus{};

      // We got a result message.
      MPI_Recv(primes.data(), jobSize, MPI_INT, workerRank, int(mpiTag::result), MPI_COMM_WORLD, &resultStatus);
      MPI_Get_count(&resultStatus, MPI_INT, &recvPrimes);

      // Convert to string.
      std::stringstream textPrimes{};
      textPrimes << "Recv block of " << recvPrimes << ": ";
      if (recvPrimes > 0)
      {
        for (int i = 0; i < recvPrimes; i++)
        {
          textPrimes << primes[i] << ", ";
        }
        textPrimes << "\n";

        MPI_File_write(outputFile, textPrimes.str().c_str(),  textPrimes.str().size(), MPI_CHAR, MPI_STATUS_IGNORE);
      }
      // Put data from result_data_buffer into a global result */
      nPrimes += recvPrimes;
    }
  }

  std::string totalPrimes = "Total number of primes:" + std::to_string(nPrimes) + "\n";
  MPI_File_write(outputFile, totalPrimes.c_str(),totalPrimes.size(), MPI_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&outputFile);
  return nPrimes;
}// end of farmer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Worker process.
 *
 * 1. Ask for a new job.
 * 2. Check for incoming message.
 * 3. If it is a job message, accept the job, process it and send back the result.
 * 4. If it is a stop message, terminate.
 */
void worker()
{
  std::vector<int> primes{};

  // Shall I stop the execution?
  bool stopped = false;
  do
  {
    MPI_Status messageTypeStatus{};
    int        jobRequestMessage = 1;

    // Send a message to the master asking for a job.
    MPI_Send(&jobRequestMessage, 1, MPI_INT, MPI_ROOT_RANK, int(mpiTag::askForJob), MPI_COMM_WORLD);

    // What type of message did I receive?
    MPI_Probe(MPI_ROOT_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &messageTypeStatus);

    if (mpiTag(messageTypeStatus.MPI_TAG) == mpiTag::jobData)
    {
      // There's a message with a new job waiting for being received.
      MPI_Status jobStatus{};
      int job[2]{};

      // Retrieve job data from master into msg_buffer
      MPI_Recv(job, 2, MPI_INT,  MPI_ROOT_RANK, int(mpiTag::jobData), MPI_COMM_WORLD , &jobStatus);

      // Work on job_data to get a result and store the result into result_buffer.
      sieve(job[0], job[1], primes);

      // Send result to master.
      MPI_Send(primes.data(), primes.size(), MPI_INT, MPI_ROOT_RANK, int(mpiTag::result), MPI_COMM_WORLD);
    }
    else
    {
      // We got a stop message we have to retrieve it by using MPI_Recv.
      // But we can ignore the data from the MPI_Recv call.
      MPI_Status stopStatus{};
      int        stopMessage{};

      MPI_Recv(&stopMessage, 1, MPI_INT, MPI_ROOT_RANK, int(mpiTag::stop), MPI_COMM_WORLD , &stopStatus);
      stopped = true;
    }
 } while (!stopped);
}// end of worker
//----------------------------------------------------------------------------------------------------------------------


/**
 * Parallel sieve.
 * @param [in] start   - Minimum tested number.
 * @param [in] stop    - Maximum tested number.
 * @param [in] jobSize - Job size.
 * @return number of primes
 */
int parSeive(const int start,
             const int stop,
             const int jobSize)
{
  int nPrimes{};

  if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
  {
    nPrimes = farmer(start, stop, jobSize);
  }
  else
  {
    worker();
  }

  return nPrimes;
}// end of parSeive
//----------------------------------------------------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                    Main routine                                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Main function
 */
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  // Set stdout to print out
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
  mpiPrintf(MPI_ROOT_RANK, "                           PPP Lab 8 - Farm                       \n");
  mpiPrintf(MPI_ROOT_RANK, "                       Count the number of Primes                 \n");
  mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
  mpiFlush();

  // Check number of parameters.
  if (argc != 3)
  {
    mpiPrintf(MPI_ROOT_RANK, "!!! Please specify maximum number and job size in thousands!   !!!\n");
    mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Parse parameters
  const int maxNumber = std::stoi(argv[1]) * 1000;
  const int jobSize   = std::stoi(argv[2]) * 1000;

  double startTime{}, stopTime{};
  double parTime{};

  int parPrimes{};

  //------------------------------------------------- PAR version ----------------------------------------------------//
  mpiPrintf(MPI_ROOT_RANK, " Par search for primes between 2 and %d\n", maxNumber);
  mpiPrintf(MPI_ROOT_RANK, " Number of processes %d.\n", mpiGetCommSize(MPI_COMM_WORLD));

  startTime = MPI_Wtime();

  parPrimes = parSeive(2, maxNumber, jobSize);

  stopTime = MPI_Wtime();
  parTime = (stopTime - startTime);

  mpiPrintf(MPI_ROOT_RANK, " Par number of primes %d\n",   parPrimes);
  mpiPrintf(MPI_ROOT_RANK, " Par execution time %7.3fs\n", parTime);
  mpiPrintf(MPI_ROOT_RANK, "------------------------------------------------------------------\n");

  MPI_Finalize();
}// end of main
//----------------------------------------------------------------------------------------------------------------------
