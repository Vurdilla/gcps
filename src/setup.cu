#ifndef VC3_PHYS_KSNP_SETUP
#define VC3_PHYS_KSNP_SETUP

#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>

#include "../include/general/stringext.cu"
#include "../include/types.cu"
#include "../include/general/INIReader.cu"


namespace vc3_phys{

/** KSNP target parameters 
**/
struct targetParameters
{
	flt2 targetDistanceMin; // minimum distance from the origin
	flt2 targetDistanceMax; // maximum distance from the origin
	flt2 targetAngleMin; // minimum anglular position, in degrees
	flt2 targetAngleMax; // maximum anglular position, in degrees
	flt2 targetRadiusMin; // minimum target radius
	flt2 targetRadiusMax; // maximum target radius
	flt2 targetWeightMin; // minimum target weight (number of first hits to capture)
	flt2 targetWeightMax; // maximum target weight (number of first hits to capture)
	int targetAppearType; // 0 - appear by time, 1 - appear after termination
	flt2 targetAppearTime; // loop appear time
	flt2 targetAppearDelay; // time delay before appear trigger and real appear of the target
	int targetTerminateType; // 0 - terminate by time, 1 - terminate after capture
	flt2 targetTerminateTime; // loop terminate time
	int targetResetType; // target reset: 
	//   0 - position to the center + random direction
	//   1 - direction to the center + one timestep towards center
	//   2 - reverse direction + one timestep backward
	//   10 - permeatable target, replacing particles v and beta with targets values while the particle is inside the target
	flt2 betaTargetMultMin; // minimum beta multiplication coefficient when particle hits target
	flt2 betaTargetMultMax; // maximum beta multiplication coefficient when particle hits target
	flt2 VTargetMin; // minimum particle velocity set when particle hits target
	flt2 VTargetMax; // maximum particle velocity set when particle hits target
	bool targetResetSRtime; // is target hit resets stochastic reset time or not
	bool targetHomePotential; // does target hit switches on home potential sensing
};

/** KSNP partile scenario parameters
**/
struct scenarioParticles
{
	int parameter; 
	/* value to change: 
	1 - particle_V0
	2 - particle_beta0
	3 - particle_chiT
	4 - particle_chiR
	5 - particle_c0
	6 - particle_DR
	*/

	std::vector<long long int> timesteppoints;
	std::vector<flt2> values;
};

/** KSNP setup data
**/
struct setupParameters
{
	// default constructor
	setupParameters();

	// Read parameters from an ini file
	__host__ int read(std::string filename);

	// Write parameters to an ini file
	__host__ int write(std::string filename, bool header = true);

	// searcher parameters
	int nParticles; // number of particles, default = 1
	int chemosensitivityModel; // chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)
	// searcher velocity, default =  1
	int V0_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	flt2 V0, V0_min, V0_max, V0_bias, V0_mean, V0_sigma;
	bool keepV0Constant; // use equations with constant V0, default = false
	flt2 VDecayTime; // searcher velocity relaxation rate, default = 1.0
					 // positive = there is a decay, zero = no decay, negative = reset to v0 every timestep
	// rotationbal diffusion coefficient, default = 1.0
	int rotationalDiffusion_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	flt2 rotationalDiffusion, rotationalDiffusion_min, rotationalDiffusion_max, rotationalDiffusion_bias, rotationalDiffusion_mean, rotationalDiffusion_sigma;
	flt2 Rscent; // scent raduis, default = 0.05
	// searcher chemodeposition rate, default = 0.005
	int beta0_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	flt2 beta0, beta0_min, beta0_max, beta0_bias, beta0_mean, beta0_sigma;
	flt2 betaDecayTime; // searcher chemodeposition relaxation rate, default = 1.0
						// positive = there is a decay, zero = no decay, negative = reset to beta0 every timestep
	// rotational chemosensitivity, default =  0.2
	int chiRot_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	flt2 chiRot, chiRot_min, chiRot_max, chiRot_bias, chiRot_mean, chiRot_sigma;
	// translational chemosensitivity, default =  0.015
	int chiTrans_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	flt2 chiTrans, chiTrans_min, chiTrans_max, chiTrans_bias, chiTrans_mean, chiTrans_sigma;
	// scent noise level, default = 1.00
	int SC0_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	flt2 SC0, SC0_min, SC0_max, SC0_bias, SC0_mean, SC0_sigma;

	// pairwise searcher interactions
	flt2 PPepsilon; // soft potential epsilon
	flt2 PPsigma; // soft potential sigma

	// system parameters
	flt2 boxSize; // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
	int boundaryType; // boundary: 0 - circle, 1 - square, 2 - square PBC
	flt2 scentDecayTime; // scent decay time, default = 1.00
	int initialNTRtype; // initian next time time reset type: 0 - same as regular, 1 - random at [0.00, timedResetMeanTime)
	int initialParticlePos; // 0 - center of arena, 1 - uniform distribution across arena area

	// home potential
	int homeType; // home type: 1 - parabolic potential, 2 - conical potential
	flt2 homeRadius; // radius of the circular area from which particles instantly resets to the center
	flt2 homePotentialKT; // strength of the home potential, translational strength
	flt2 homePotentialKR; // strength of the home potential, rotational strength

	// timed reset parameters
	int timedResetTimerType; // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
	int timedResetType; // timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
	//   2 - reverse direction + one timestep backward
	flt2 timedResetMeanTime; // average time between resets, default = 50
	bool timedResetHomePotential; // does timed reset switches on home potential sensing
	flt2 globalResetTime; // time marker for all-particle reset, default -1 (not set), active when non-negative

	// boundary parameters
	int boundaryResetType; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
	//   2 - reverse direction + one timestep backward
	flt2 betaBoundaryMult; // beta multiplication coefficient when particle hits boundary
	flt2 VBoundary; // velocity when particle hits boundary
	bool boundaryResetSRtime; // does boundary hit reset stochastic reset time or not
	bool boundaryHomePotential; // does boundary hit switches on home potential sensing

	// target parameters
	std::vector<targetParameters> targetList;

	// computation parameters
	int latticeSize; // number of nodes on each side of the grid, default = 200
	flt2 timeStep; // length of time step, default = 0.005
	flt2 Rscent_cutoffMultiplier; // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
	long long int scentDecayRescalingNsteps; // number of steps between inflation rescaling, defaul = 100000
	int gpuHistoryLength; // number of steps stored on GPU before it's flushed to CPU and analyzed

	// run parameters
	long long int nSteps; // total number of time steps in each iteration, default = 20000
	int nIterations; // total number of iterations, default = 100

	// scenario parameters
	std::vector<scenarioParticles> scenarioParticlesList;

	// output parameters
	long long int cumulativeDataMinStep; // minimum step after which all cumulative data to be collected
	int cumulativePCblockSize; // block size for the cumulative particle concentration
	int cumulativeRnbins; // cumulativeRnbins
	long long int PCReg_startstep;
	long long int PCReg_window;
	int PCReg_blockSize; // block size for the cumulative particle concentration
	long long int SCReg_startstep;
	long long int SCReg_window;
	int SCReg_blockSize; // block size for the cumulative particle concentration
	//flt2 stepDrHistogram_min, stepDrHistogram_max; // <-- * NOT USED * // limits of dtep dr histogram
	//int stepDrHistogram_nbins; // <-- * NOT USED * // number of bins in dtep dr histogram
	//long long int totalScentReg_window; // <-- * NOT USED * // regular time total scent will be averaged inside window in each iteration, in steps
	//long long int totalScentReg_avgperiod; // <-- * NOT USED * // regular time total scent will be averaged with this period in each iteration, in steps
	//int PCrtheta_r_nbins; // <-- * NOT USED * // number of bins in rtheta PC hist over radius
	//flt2 PCrtheta_theta_binsize; // <-- * NOT USED * // linear size of bins in rtheta PC hist over theta
	//long long int PCrthetaReg_window; // <-- * NOT USED * // regular time rtheta PC will be accumulated inside window in each iteration, in steps
	//int PCtheta_nbins; // <-- * NOT USED * // number of bins in theta PC hist
	//flt2 PCtheta_rmin; // <-- * NOT USED * // r threshold after which particles participates in theta PC hist
	//flt2 PCtheta_rmax;  // <-- * NOT USED *// r threshold after which particles participates in theta PC hist
	//long long int PCthetaReg_window; // <-- * NOT USED * // regular time theta PC will be accumulated inside window in each iteration, in steps
	long long int HitCountReg_window; // regular time interval at which hit counts are stored, in steps
	long long int HitCountReg_minStep; // minimum step after which all hit counts are on, in steps 
	bool HitCountPP_afterReset; // does per-particle hit count counts only after the first reset
	bool printTrjs; // print trajectories flag
	int trjStepFrom; // print trajectories from this step
	int trjStepTo; // print trajectories to this step (negative = until the end of sim)
	int trjBlockEverySteps;
	int trjBlockDurationSteps;
	int trjBlockFreqSteps;
	int trjFormat; // trajectories format: 0 - LAMMPS x y vx vy [particle properties]
	std::string trjParticleProperties; // options are: V0, beta0, chiT, chiR, c0, DR
	int trjPrecision; // precision of numerical data in trj output: 0 - default (6 digits), 1 - extended (10 digits), 2 - full
	//bool computeMSD; // <-- * NOT USED * // Compute MSD of the trajectories
	//int maxMSDLength; // <-- * NOT USED * // length of MSD output range
	//int MSDstep; // <-- * NOT USED * // step count between the nearest MSD points
	//flt2 PRH_min; // <-- * NOT USED * // limits of path-rotation histogram
	//flt2 PRH_max; // <-- * NOT USED * // limits of path-rotation histogram
	//int PRH_nbins; // <-- * NOT USED * // number of bins in path-rotation histogram
	//int PRH_step; // <-- * NOT USED * // step between the nearest points for PRH
	//int GTFPT_blockSize; // <-- * NOT USED * // block size for GT FPT collection
	//std::vector<flt2> GTFPT_Rtarget; // <-- * NOT USED * // GT FPT target radius vector

}; // struct setupParameters

} //namespace vc3_phys


vc3_phys::setupParameters::setupParameters()
{
	// searcher parameters
	nParticles = 1; // number of particles, default = 1
	chemosensitivityModel = 0; // chemosensitivity model : 0 - ~gradient of SC, 1 - gradient of log(SC)
	// searcher velocity, default =  1
	V0_distr = 0;
	V0 = 1; V0_min = V0_max = V0_mean = V0_sigma = 0; V0_bias = 0.5;
	keepV0Constant = false; // use equations with constant V0, default = false
	VDecayTime = 1.0; // searcher velocity relaxation rate, default = 1.0
	// rotationbal diffusion coefficient, default = 1.0
	rotationalDiffusion_distr = 0;
	rotationalDiffusion = 1.0; rotationalDiffusion_min = rotationalDiffusion_max = rotationalDiffusion_mean = rotationalDiffusion_sigma = 0; rotationalDiffusion_bias = 0.5;
	Rscent = 0.05; // scent raduis, default = 0.05
	// searcher chemodeposition rate, default = 0.005
	beta0_distr = 0;
	beta0 = 0.005; beta0_min = beta0_max = beta0_mean = beta0_sigma = 0; beta0_bias = 0.5;
	// searcher chemodeposition relaxation rate, default = 1.0
	betaDecayTime = 1.0;
	// rotational chemosensitivity, default =  0.2
	chiRot_distr = 0;
	chiRot = 0.2; chiRot_min = chiRot_max = chiRot_mean = chiRot_sigma = 0; chiRot_bias = 0.5;
	// translational chemosensitivity, default =  0.015
	chiTrans_distr = 0;
	chiTrans = 0.015; chiTrans_min = chiTrans_max = chiTrans_mean = chiTrans_sigma = 0; chiTrans_bias = 0.5;
	// scent noise level, default = 1.00
	SC0_distr = 0;
	SC0 = 1.00; SC0_min = SC0_max = SC0_mean = SC0_sigma = 0; SC0_bias = 0.5;

	// pairwise searcher interactions
	PPepsilon = 0.00; // soft potential epsilon
	PPsigma = 0.01; // soft potential sigma

	// system parameters
	boxSize = 1.00; // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
	boundaryType = 0; // boundary: 0 - circle, 1 - square, 2 - square PBC
	scentDecayTime = 1.00; // scent decay time, default = 1.00
	initialNTRtype = 0; // initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)
	initialParticlePos = 0; // 0 - center of arena, 1 - uniform distribution across arena area

	// home potential
	homeType = 0; // home type: 1 - parabolic potential, 2 - conical potential
	homeRadius = 0.1; // radius of the circular area at the origin called 'home'
	homePotentialKT = 0.0; // Strength of the home potential, translational strength
	homePotentialKR = 0.0; // Strength of the home potential, rotational strength

	// timed reset parameters
	timedResetTimerType = 0; // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
	timedResetType = 0; // stochastic reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
	//   2 - reverse direction + one timestep backward
	timedResetMeanTime = 50; // average time between resets, default = 50
	timedResetHomePotential = false; // does timed reset switches on home potential sensing
	globalResetTime = -1; // time marker for all-particle reset, default -1 (not set), active when non-negative

	// boundary parameter	
	boundaryResetType = 0; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center
	//   2 - reverse direction + one timestep towards center
	betaBoundaryMult = 0.00; // beta multiplication coefficient when particle hits boundary
	VBoundary = 0.00; // velocity when particle hits boundary
	boundaryResetSRtime = true; // is boundary hit resets stochastic reset time or not
	boundaryHomePotential = false; // does boundary hit switches on home potential sensing

	// target parameters

	// computation parameters
	latticeSize = 200; // number of nodes on each side of the grid, default = 200
	timeStep = 0.005; // length of time step, default = 0.005
	Rscent_cutoffMultiplier = 3.0; // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
	scentDecayRescalingNsteps = 100000; // number of steps between inflation rescaling, default = 100000

	// run parameters
	nSteps = 10000; // total number of time steps in each iteration, default = 10000
	nIterations = 1; // total number of iterations, default = 1

	// scenario parameters

	// output parameters
	cumulativeDataMinStep = 0;
	cumulativePCblockSize = 1;
	cumulativeRnbins = 1;
	/*stepDrHistogram_min = 0.00; // limits of step dr histogram
	stepDrHistogram_max = 0.02; // limits of step dr histogram
	stepDrHistogram_nbins = 400; // number of bins in step dr histogram
	totalScentReg_window = 100000; // regular time total scent will be averaged inside window in each iteration, in steps
	totalScentReg_avgperiod = 1000000; // regular time total scent will be averaged with this period in each iteration, in steps
	PCrtheta_r_nbins = 10; // number of bins in rtheta particle concentration hist over radius
	PCrtheta_theta_binsize = 1.00; // linear size of bins in rtheta particle concentration hist over theta
	PCrthetaReg_window = 1000000; // regular time rtheta PC will be accumulated inside window in each iteration, in steps
	PCtheta_nbins = 10; // number of bins in theta PC hist
	PCtheta_rmin = 0.00; // r threshold after which particles participates in theta PC hist
	PCtheta_rmax = 0.00; // r threshold after which particles participates in theta PC hist
	PCthetaReg_window = 1000000; // regular time theta PC will be accumulated inside window in each iteration, in steps*/
	HitCountReg_window = 100000; // regular time interval at which hit counts are stored, in steps
	HitCountReg_minStep = 0; // minimum step after which all hit counts are on, in steps 
	HitCountPP_afterReset = false; // does per-particle hit count counts only after the first reset
	printTrjs = false; // print trajectories flag
	trjStepFrom = 0; // print trajectories from this step
	trjStepTo = -1; // print trajectories to this step (negative = until the end of sim)
	trjBlockEverySteps = 10; 
	trjBlockDurationSteps = 10;
	trjBlockFreqSteps = 1;
	trjFormat = 0; // 0 - LAMMPS x y vx vy
	trjParticleProperties = "";
	trjPrecision = 0; // precision of numerical data in trj output: 0 - default (6 digits), 1 - extended (10 digits), 2 - full
	/*NTrj = 0;
	EveryTrj = 1;
	computeMSD = false;
	maxMSDLength = 1;
	MSDstep = 1;
	PRH_min = 0.00;
	PRH_max = 1.00;
	PRH_nbins = 1;
	PRH_step = 1;
	GTFPT_blockSize = 1;
	GTFPT_Rtarget.resize(0);*/
}

__host__ int vc3_phys::setupParameters::read(std::string filename)
{
	INIReader ini(filename);
	switch (ini.ParseError())
	{
	case 0: // parsed successfully
		break;
	default:
		std::cout << "Error parsing parameters file \"" << filename << "\"\n";
		return 1;
	}

	// searcher parameters
	std::string distr;
	nParticles = ini.GetInteger("Searcher", "nParticles", 1); // number of particles, default = 1
	chemosensitivityModel = ini.GetInteger("Searcher", "chemosensitivityModel", 1.0); // chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)
	// searcher velocity, default =  1
	distr = ini.Get("Searcher", "V0_distr", "single");
	if (distr == "bimodal") V0_distr = 1;
	else if (distr == "uniform") V0_distr = 2;
	else if (distr == "log-uniform") V0_distr = 3;
	else if (distr == "gaussian") V0_distr = 4;
	else if (distr == "log-gaussian") V0_distr = 5;
	else V0_distr = 0;
	switch (V0_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
	case 2:
	case 3:
		V0_min = ini.GetReal("Searcher", "V0_min", 1.0);
		V0_max = ini.GetReal("Searcher", "V0_max", 2.0);
		V0_bias = ini.GetReal("Searcher", "V0_bias", 0.5);
		break;
	case 4:
	case 5:
		V0_min = ini.GetReal("Searcher", "V0_min", 1.0);
		V0_max = ini.GetReal("Searcher", "V0_max", 2.0);
		V0_mean = ini.GetReal("Searcher", "V0_mean", 1.0);
		V0_sigma = ini.GetReal("Searcher", "V0_sigma", 1.0);
		break;
	case 0:
	default:
		V0 = ini.GetReal("Searcher", "V0", 1.0);
	}
	keepV0Constant = ini.GetBoolean("Searcher", "keepV0Constant", false); // use equations with constant V0, default = false
	VDecayTime = ini.GetReal("Searcher", "VDecayTime", 1.0);
	// rotationbal diffusion coefficient, default = 1.0
	distr = ini.Get("Searcher", "rotationalDiffusion_distr", "single");
	if (distr == "bimodal") rotationalDiffusion_distr = 1;
	else if (distr == "uniform") rotationalDiffusion_distr = 2;
	else if (distr == "log-uniform") rotationalDiffusion_distr = 3;
	else if (distr == "gaussian") rotationalDiffusion_distr = 4;
	else if (distr == "log-gaussian") rotationalDiffusion_distr = 5;
	else rotationalDiffusion_distr = 0;
	switch (rotationalDiffusion_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
	case 2:
	case 3:
		rotationalDiffusion_min = ini.GetReal("Searcher", "rotationalDiffusion_min", 1.0);
		rotationalDiffusion_max = ini.GetReal("Searcher", "rotationalDiffusion_max", 2.0);
		rotationalDiffusion_bias = ini.GetReal("Searcher", "rotationalDiffusion_bias", 0.5);
		break;
	case 4:
	case 5:
		rotationalDiffusion_min = ini.GetReal("Searcher", "rotationalDiffusion_min", 1.0);
		rotationalDiffusion_max = ini.GetReal("Searcher", "rotationalDiffusion_max", 2.0);
		rotationalDiffusion_mean = ini.GetReal("Searcher", "rotationalDiffusion_mean", 1.0);
		rotationalDiffusion_sigma = ini.GetReal("Searcher", "rotationalDiffusion_sigma", 1.0);
		break;
	case 0:
	default:
		rotationalDiffusion = ini.GetReal("Searcher", "rotationalDiffusion", 1.0);
	}
	Rscent = ini.GetReal("Searcher", "Rscent", 0.05); // scent raduis, default = 0.05
	// searcher chemodeposition rate, default = 0.005
	distr = ini.Get("Searcher", "beta0_distr", "single");
	if (distr == "bimodal") beta0_distr = 1;
	else if (distr == "uniform") beta0_distr = 2;
	else if (distr == "log-uniform") beta0_distr = 3;
	else if (distr == "gaussian") beta0_distr = 4;
	else if (distr == "log-gaussian") beta0_distr = 5;
	else beta0_distr = 0;
	switch (beta0_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
	case 2:
	case 3:
		beta0_min = ini.GetReal("Searcher", "beta0_min", 1.0);
		beta0_max = ini.GetReal("Searcher", "beta0_max", 2.0);
		beta0_bias = ini.GetReal("Searcher", "beta0_bias", 0.5);
		break;
	case 4:
	case 5:
		beta0_min = ini.GetReal("Searcher", "beta0_min", 1.0);
		beta0_max = ini.GetReal("Searcher", "beta0_max", 2.0);
		beta0_mean = ini.GetReal("Searcher", "beta0_mean", 1.0);
		beta0_sigma = ini.GetReal("Searcher", "beta0_sigma", 1.0);
		break;
	case 0:
	default:
		beta0 = ini.GetReal("Searcher", "beta0", 0.005);
	}
	// searcher chemodeposition relaxation rate, default = 1.0
	betaDecayTime = ini.GetReal("Searcher", "betaDecayTime", 1.0);
	// rotational chemosensitivity, default =  0.2
	distr = ini.Get("Searcher", "chiRot_distr", "single");
	if (distr == "bimodal") chiRot_distr = 1;
	else if (distr == "uniform") chiRot_distr = 2;
	else if (distr == "log-uniform") chiRot_distr = 3;
	else if (distr == "gaussian") chiRot_distr = 4;
	else if (distr == "log-gaussian") chiRot_distr = 5;
	else chiRot_distr = 0;
	switch (chiRot_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
	case 2:
	case 3:
		chiRot_min = ini.GetReal("Searcher", "chiRot_min", 1.0);
		chiRot_max = ini.GetReal("Searcher", "chiRot_max", 2.0);
		chiRot_bias = ini.GetReal("Searcher", "chiRot_bias", 0.5);
		break;
	case 4:
	case 5:
		chiRot_min = ini.GetReal("Searcher", "chiRot_min", 1.0);
		chiRot_max = ini.GetReal("Searcher", "chiRot_max", 2.0);
		chiRot_mean = ini.GetReal("Searcher", "chiRot_mean", 1.0);
		chiRot_sigma = ini.GetReal("Searcher", "chiRot_sigma", 1.0);
		break;
	case 0:
	default:
		chiRot = ini.GetReal("Searcher", "chiRot", 0.2);
	}
	// translational chemosensitivity, default =  0.015
	distr = ini.Get("Searcher", "chiTrans_distr", "single");
	if (distr == "bimodal") chiTrans_distr = 1;
	else if (distr == "uniform") chiTrans_distr = 2;
	else if (distr == "log-uniform") chiTrans_distr = 3;
	else if (distr == "gaussian") chiTrans_distr = 4;
	else if (distr == "log-gaussian") chiTrans_distr = 5;
	else chiTrans_distr = 0;
	switch (chiTrans_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
	case 2:
	case 3:
		chiTrans_min = ini.GetReal("Searcher", "chiTrans_min", 1.0);
		chiTrans_max = ini.GetReal("Searcher", "chiTrans_max", 2.0);
		chiTrans_bias = ini.GetReal("Searcher", "chiTrans_bias", 0.5);
		break;
	case 4:
	case 5:
		chiTrans_min = ini.GetReal("Searcher", "chiTrans_min", 1.0);
		chiTrans_max = ini.GetReal("Searcher", "chiTrans_max", 2.0);
		chiTrans_mean = ini.GetReal("Searcher", "chiTrans_mean", 1.0);
		chiTrans_sigma = ini.GetReal("Searcher", "chiTrans_sigma", 1.0);
		break;
	case 0:
	default:
		chiTrans = ini.GetReal("Searcher", "chiTrans", 0.015);
	}
	// scent noise level, default = 1.00
	distr = ini.Get("Searcher", "SC0_distr", "single");
	if (distr == "bimodal") SC0_distr = 1;
	else if (distr == "uniform") SC0_distr = 2;
	else if (distr == "log-uniform") SC0_distr = 3;
	else if (distr == "gaussian") SC0_distr = 4;
	else if (distr == "log-gaussian") SC0_distr = 5;
	else SC0_distr = 0;
	switch (SC0_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
	case 2:
	case 3:
		SC0_min = ini.GetReal("Searcher", "SC0_min", 1.0);
		SC0_max = ini.GetReal("Searcher", "SC0_max", 2.0);
		SC0_bias = ini.GetReal("Searcher", "SC0_bias", 0.5);
		break;
	case 4:
	case 5:
		SC0_min = ini.GetReal("Searcher", "SC0_min", 1.0);
		SC0_max = ini.GetReal("Searcher", "SC0_max", 2.0);
		SC0_mean = ini.GetReal("Searcher", "SC0_mean", 1.0);
		SC0_sigma = ini.GetReal("Searcher", "SC0_sigma", 1.0);
		break;
	case 0:
	default:
		SC0 = ini.GetReal("Searcher", "SC0", 1.0);
	}

	// pairwise searcher interactions
	PPepsilon = ini.GetReal("Pairwise", "PPepsilon", 0.00); // soft potential epsilon
	PPsigma = ini.GetReal("Pairwise", "PPsigma", 0.01); // soft potential sigma

	// system parameters
	boxSize = ini.GetReal("System", "boxSize", 1.0); // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
	boundaryType = ini.GetInteger("System", "boundaryType", 0); // boundary: 0 - circle, 1 - square, 2 - square PBC
	scentDecayTime = ini.GetReal("System", "scentDecayTime", 1.0); // scent decay time, default = 1.00
	initialNTRtype = ini.GetReal("System", "initialNTRtype", 0); // initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)
	initialParticlePos = ini.GetReal("System", "initialParticlePos", 0); // 0 - center of arena, 1 - uniform distribution across arena area

	// home potential
	homeType = ini.GetInteger("Home", "homeType", 0); // home type: 1 - parabolic potential, 2 - conical potential
	homeRadius = ini.GetReal("Home", "homeRadius", 0.1); // radius of the circular area at the origin called 'home'
	homePotentialKT = ini.GetReal("Home", "homePotentialKT", 0.0); // strength of the home potential, translational strength
	homePotentialKR = ini.GetReal("Home", "homePotentialKR", 0.0); // strength of the home potential, rotational strength

	// timed reset parameters
	timedResetTimerType = ini.GetInteger("TimedReset", "timedResetTimerType", 0); // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
	timedResetType = ini.GetInteger("TimedReset", "timedResetType", 0); // timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward
	timedResetMeanTime = ini.GetReal("TimedReset", "timedResetMeanTime", 50.0); // average time between timed resets, default =  50
	timedResetHomePotential = ini.GetBoolean("TimedReset", "timedResetHomePotential", false); // does timed reset switches on home potential sensing
	globalResetTime = ini.GetReal("TimedReset", "globalResetTime", -1.0); // time marker for all-particle reset, default -1 (not set), active when non-negative

	// boundary parameter	
	boundaryResetType = ini.GetInteger("Boundary", "boundaryResetType", 0);; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep towards center
	betaBoundaryMult = ini.GetReal("Boundary", "betaBoundaryMult", 1.0); // beta multiplication coefficient when particle hits boundary
	VBoundary = ini.GetReal("Boundary", "VBoundary", 1.0); // velocity when particle hits boundary
	boundaryResetSRtime = ini.GetBoolean("Boundary", "boundaryResetSRtime", false); // is boundary hit resets stochastic reset time or not
	boundaryHomePotential = ini.GetBoolean("Boundary", "boundaryHomePotential", false); // does boundary hit switches on home potential sensing

	// target list parameters
	int ntargets = ini.GetInteger("TargetList", "targetNtargets", 0);
	targetList.resize(ntargets);
	for (int nt = 0; nt < ntargets; nt++)
	{
		std::string tname = "Target" + vc3_general::itoa(nt + 1);
		targetList[nt].targetDistanceMin = ini.GetReal(tname, "targetDistanceMin", 0.00);
		targetList[nt].targetDistanceMax = ini.GetReal(tname, "targetDistanceMax", 0.00);
		targetList[nt].targetAngleMin = ini.GetReal(tname, "targetAngleMin", 0.00);
		targetList[nt].targetAngleMax = ini.GetReal(tname, "targetAngleMax", 0.00);
		targetList[nt].targetRadiusMin = ini.GetReal(tname, "targetRadiusMin", 0.00);
		targetList[nt].targetRadiusMax = ini.GetReal(tname, "targetRadiusMax", 0.00);
		targetList[nt].targetWeightMin = ini.GetReal(tname, "targetWeightMin", 0.00);
		targetList[nt].targetWeightMax = ini.GetReal(tname, "targetWeightMax", 0.00);
		targetList[nt].targetAppearType = ini.GetInteger(tname, "targetAppearType", 0);
		targetList[nt].targetAppearTime = ini.GetReal(tname, "targetAppearTime", 0);
		targetList[nt].targetAppearDelay = ini.GetReal(tname, "targetAppearDelay", 0.00);
		targetList[nt].targetTerminateType = ini.GetInteger(tname, "targetTerminateType", 0.00);
		targetList[nt].targetTerminateTime = ini.GetReal(tname, "targetTerminateTime", 0.00);
		targetList[nt].targetResetType = ini.GetInteger(tname, "targetResetType", 0.00);
		targetList[nt].betaTargetMultMin = ini.GetReal(tname, "betaTargetMultMin", 1.00);
		targetList[nt].betaTargetMultMax = ini.GetReal(tname, "betaTargetMultMax", 1.00);
		targetList[nt].VTargetMin = ini.GetReal(tname, "VTargetMin", -1.00);
		targetList[nt].VTargetMax = ini.GetReal(tname, "VTargetMax", -1.00);
		targetList[nt].targetResetSRtime = ini.GetBoolean(tname, "targetResetSRtime", true);
		targetList[nt].targetHomePotential = ini.GetBoolean(tname, "targetHomePotential", false);
	}

	// computations parameters
	latticeSize = ini.GetInteger("Computations", "latticeSize", 200); // number of nodes on each side of the grid, default = 200
	timeStep = ini.GetReal("Computations", "timeStep", 0.005); // length of time step, default = 0.005
	Rscent_cutoffMultiplier = ini.GetReal("Computations", "Rscent_cutoffMultiplier", 3.0); // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
	scentDecayRescalingNsteps = ini.GetLLInteger("Computations", "scentDecayRescalingNsteps", 100000); // number of steps between inflation rescaling, default = 100000
	gpuHistoryLength = ini.GetInteger("Computations", "gpuHistoryLength", 1000); // number of steps stored on GPU before it's flushed to CPU and analyzed

	// run parameters
	nSteps = ini.GetLLInteger("Run", "nSteps", 10000); // total number of time steps in each iteration, default = 10000
	nIterations = ini.GetInteger("Run", "nIterations", 1); // total number of iterations, default = 1

	// scenario parameters
	int nscenarioparticles = ini.GetInteger("Scenarios", "scenarioParticlesNscenarios", 0);
	scenarioParticlesList.resize(nscenarioparticles);
	for (int ns = 0; ns < nscenarioparticles; ns++)
	{
		std::string sname = "scenarioParticles" + vc3_general::itoa(ns + 1);
		// parameter
		std::string parname = ini.Get(sname, "scenarioParticlesParameter", "");
		scenarioParticlesList[ns].parameter = 0;
		if (parname == "particle_V0") scenarioParticlesList[ns].parameter = 1;
		else if (parname == "particle_beta0") scenarioParticlesList[ns].parameter = 2;
		else if (parname == "particle_chiT") scenarioParticlesList[ns].parameter = 3;
		else if (parname == "particle_chiR") scenarioParticlesList[ns].parameter = 4;
		else if (parname == "particle_c0") scenarioParticlesList[ns].parameter = 5;
		else if (parname == "particle_DR") scenarioParticlesList[ns].parameter = 6;
		// timesteppoints
		scenarioParticlesList[ns].timesteppoints.clear();
		std::string timestepstring = ini.Get(sname, "scenarioParticlesTimesteppoints", "");
		std::replace(timestepstring.begin(), timestepstring.end(), ',', ' ');
		std::replace(timestepstring.begin(), timestepstring.end(), ';', ' ');
		std::stringstream stimestepstring(timestepstring);
		long long int timestep;
		while (stimestepstring >> timestep)
			scenarioParticlesList[ns].timesteppoints.push_back(timestep);
		// values
		scenarioParticlesList[ns].values.clear();
		std::string valuestring = ini.Get(sname, "scenarioParticlesValues", "");
		std::replace(valuestring.begin(), valuestring.end(), ',', ' ');
		std::replace(valuestring.begin(), valuestring.end(), ';', ' ');
		std::stringstream svaluestring(valuestring);
		flt2 value;
		while (svaluestring >> value)
			scenarioParticlesList[ns].values.push_back(value);
		if (scenarioParticlesList[ns].values.size() != scenarioParticlesList[ns].timesteppoints.size())
		{
			std::cout << "\nParameter file \"" << filename << "\" SCENARIO " << sname << ": timesteppoints and values have different sizes!\n";
			std::cout.flush();
			return 2;
		}
	}

	// output parameters
	cumulativeDataMinStep = ini.GetInteger("Output", "cumulativeDataMinStep", 0);
	cumulativePCblockSize = ini.GetInteger("Output", "cumulativePCblockSize", 1);
	cumulativeRnbins = ini.GetInteger("Output", "cumulativeRnbins", 1);
	PCReg_startstep = ini.GetLLInteger("Output", "PCReg_startstep", 0);
	PCReg_window = ini.GetLLInteger("Output", "PCReg_window", 0);
	PCReg_blockSize = ini.GetInteger("Output", "PCReg_blockSize", 1);
	SCReg_startstep = ini.GetLLInteger("Output", "SCReg_startstep", 0);
	SCReg_window = ini.GetLLInteger("Output", "SCReg_window", 0);
	SCReg_blockSize = ini.GetInteger("Output", "SCReg_blockSize", 1);
	/*stepDrHistogram_min = ini.GetReal("Output", "stepDrHistogram_min", 0);
	stepDrHistogram_max = ini.GetReal("Output", "stepDrHistogram_max", 0);
	stepDrHistogram_nbins = ini.GetInteger("Output", "stepDrHistogram_nbins", 0);
	totalScentReg_window = ini.GetLLInteger("Output", "totalScentReg_window", 0);
	totalScentReg_avgperiod = ini.GetLLInteger("Output", "totalScentReg_avgperiod", 0);
	PCrtheta_r_nbins = ini.GetInteger("Output", "PCrtheta_r_nbins", 10);
	PCrtheta_theta_binsize = ini.GetReal("Output", "PCrtheta_theta_binsize", 1.00);
	PCrthetaReg_window = ini.GetLLInteger("Output", "PCrthetaReg_window", 0);
	PCtheta_nbins = ini.GetInteger("Output", "PCtheta_nbins", 0);
	PCtheta_rmin = ini.GetReal("Output", "PCtheta_rmin", 0);
	PCtheta_rmax = ini.GetReal("Output", "PCtheta_rmax", 0);
	PCthetaReg_window = ini.GetLLInteger("Output", "PCthetaReg_window", 0);*/
	HitCountReg_window = ini.GetLLInteger("Output", "HitCountReg_window", 0);
	HitCountReg_minStep = ini.GetLLInteger("Output", "HitCountReg_minStep", 0);
	HitCountPP_afterReset = ini.GetBoolean("Output", "HitCountPP_afterReset", false);
	printTrjs = ini.GetBoolean("Output", "printTrjs", false);
	trjStepFrom = ini.GetInteger("Output", "trjStepFrom", 0);
	trjStepTo = ini.GetInteger("Output", "trjStepTo", -1);
	trjBlockEverySteps = ini.GetInteger("Output", "trjBlockEverySteps", 1);
	trjBlockDurationSteps = ini.GetInteger("Output", "trjBlockDurationSteps", 1);
	trjBlockFreqSteps = ini.GetInteger("Output", "trjBlockFreqSteps", 1);
	trjFormat = ini.GetInteger("Output", "trjFormat", 0);
	trjParticleProperties = ini.Get("Output", "trjParticleProperties", "");
	trjPrecision = ini.GetInteger("Output", "trjPrecision", 0);
	/*computeMSD = ini.GetBoolean("Output", "computeMSD", false);
	maxMSDLength = ini.GetInteger("Output", "maxMSDLength", 1);
	MSDstep = ini.GetInteger("Output", "MSDstep", 1);
	PRH_min = ini.GetReal("Output", "PRH_min", 0.0);
	PRH_max = ini.GetReal("Output", "PRH_max", 1.0);
	PRH_nbins = ini.GetInteger("Output", "PRH_nbins", 1);
	PRH_step = ini.GetInteger("Output", "PRH_step", 1);
	GTFPT_blockSize = ini.GetInteger("Output", "GTFPT_blockSize", 1);
	std::string s = ini.Get("Output", "GTFPT_Rtarget", "");
	vc3_general::stodv(s, &GTFPT_Rtarget, ',');*/

	std::cout << "\nParameter file \"" << filename << "\" read successfully\n";
	return 0;
}

__host__ int vc3_phys::setupParameters::write(std::string filename, bool header)
{
	std::ofstream f(filename.c_str());
	if (!f)
	{
		std::cout << "\nCan not write parameter file \"" << filename << "\"\n";
		return 1;
	}

	if (header) f << "; GCPS-cuda-v7.10 input parameters\n";
	f << "\n";
	f << "[Searcher]\n";
	f << "nParticles = " << nParticles << "\t; number of particles, default = 1\n";
	f << "chemosensitivityModel = " << chemosensitivityModel << "\t; chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)\n";
	// searcher velocity
	switch (V0_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
		f << "V0_distr = bimodal\t; searcher velocity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "V0_min = " << V0_min << "\t; min\n";
		f << "V0_max = " << V0_max << "\t; max\n";
		f << "V0_bias = " << V0_bias << "\t; bias\n";
		break;
	case 2:
		f << "V0_distr = uniform\t; searcher velocity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "V0_min = " << V0_min << "\t; min\n";
		f << "V0_max = " << V0_max << "\t; max\n";
		f << "V0_bias = " << V0_bias << "\t; bias\n";
		break;
	case 3:
		f << "V0_distr = log-uniform\t; searcher velocity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "V0_min = " << V0_min << "\t; min\n";
		f << "V0_max = " << V0_max << "\t; max\n";
		f << "V0_bias = " << V0_bias << "\t; bias\n";
		break;
	case 4:
		f << "V0_distr = gaussian\t; searcher velocity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "V0_min = " << V0_min << "\t; min\n";
		f << "V0_max = " << V0_max << "\t; max\n";
		f << "V0_mean = " << V0_mean << "\t; mean\n";
		f << "V0_sigma = " << V0_sigma << "\t; sigma\n";
		break;
	case 5:
		f << "V0_distr = log-gaussian\t; searcher velocity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "V0_min = " << V0_min << "\t; min\n";
		f << "V0_max = " << V0_max << "\t; max\n";
		f << "V0_mean = " << V0_mean << "\t; mean\n";
		f << "V0_sigma = " << V0_sigma << "\t; sigma\n";
		break;
	case 0:
	default:
		f << "V0 = " << V0 << "\t; searcher velocity\n";
	}
	f << "keepV0Constant = " << keepV0Constant << "\t; use equations with constant V0\n";
	f << "VDecayTime = " << VDecayTime << "\t; searcher velocity relaxation rate, default = 1.0\n";
	// rotationbal diffusion coefficient
	switch (rotationalDiffusion_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
		f << "rotationalDiffusion_distr = bimodal\t; rotationbal diffusion coefficient: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "rotationalDiffusion_min = " << rotationalDiffusion_min << "\t; min\n";
		f << "rotationalDiffusion_max = " << rotationalDiffusion_max << "\t; max\n";
		f << "rotationalDiffusion_bias = " << rotationalDiffusion_bias << "\t; bias\n";
		break;
	case 2:
		f << "rotationalDiffusion_distr = uniform\t; rotationbal diffusion coefficient: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "rotationalDiffusion_min = " << rotationalDiffusion_min << "\t; min\n";
		f << "rotationalDiffusion_max = " << rotationalDiffusion_max << "\t; max\n";
		f << "rotationalDiffusion_bias = " << rotationalDiffusion_bias << "\t; bias\n";
		break;
	case 3:
		f << "rotationalDiffusion_distr = log-uniform\t; rotationbal diffusion coefficient: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "rotationalDiffusion_min = " << rotationalDiffusion_min << "\t; min\n";
		f << "rotationalDiffusion_max = " << rotationalDiffusion_max << "\t; max\n";
		f << "rotationalDiffusion_bias = " << rotationalDiffusion_bias << "\t; bias\n";
		break;
	case 4:
		f << "rotationalDiffusion_distr = gaussian\t; rotationbal diffusion coefficient: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "rotationalDiffusion_min = " << rotationalDiffusion_min << "\t; min\n";
		f << "rotationalDiffusion_max = " << rotationalDiffusion_max << "\t; max\n";
		f << "rotationalDiffusion_mean = " << rotationalDiffusion_mean << "\t; mean\n";
		f << "rotationalDiffusion_sigma = " << rotationalDiffusion_sigma << "\t; sigma\n";
		break;
	case 5:
		f << "rotationalDiffusion_distr = log-gaussian\t; rotationbal diffusion coefficient: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "rotationalDiffusion_min = " << rotationalDiffusion_min << "\t; min\n";
		f << "rotationalDiffusion_max = " << rotationalDiffusion_max << "\t; max\n";
		f << "rotationalDiffusion_mean = " << rotationalDiffusion_mean << "\t; mean\n";
		f << "rotationalDiffusion_sigma = " << rotationalDiffusion_sigma << "\t; sigma\n";
		break;
	case 0:
	default:
		f << "rotationalDiffusion = " << rotationalDiffusion << "\t; rotationbal diffusion coefficient\n";
	}
	f << "Rscent = " << Rscent << "\t; scent raduis\n";
	// searcher chemodeposition rate
	switch (beta0_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
		f << "beta0_distr = bimodal\t; default scent secretion: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "beta0_min = " << beta0_min << "\t; min\n";
		f << "beta0_max = " << beta0_max << "\t; max\n";
		f << "beta0_bias = " << beta0_bias << "\t; bias\n";
		break;
	case 2:
		f << "beta0_distr = uniform\t; default scent secretion: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "beta0_min = " << beta0_min << "\t; min\n";
		f << "beta0_max = " << beta0_max << "\t; max\n";
		f << "beta0_bias = " << beta0_bias << "\t; bias\n";
		break;
	case 3:
		f << "beta0_distr = log-uniform\t; default scent secretion: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "beta0_min = " << beta0_min << "\t; min\n";
		f << "beta0_max = " << beta0_max << "\t; max\n";
		f << "beta0_bias = " << beta0_bias << "\t; bias\n";
		break;
	case 4:
		f << "beta0_distr = gaussian\t; default scent secretion: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "beta0_min = " << beta0_min << "\t; min\n";
		f << "beta0_max = " << beta0_max << "\t; max\n";
		f << "beta0_mean = " << beta0_mean << "\t; mean\n";
		f << "beta0_sigma = " << beta0_sigma << "\t; sigma\n";
		break;
	case 5:
		f << "beta0_distr = log-gaussian\t; default scent secretion: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "beta0_min = " << beta0_min << "\t; min\n";
		f << "beta0_max = " << beta0_max << "\t; max\n";
		f << "beta0_mean = " << beta0_mean << "\t; mean\n";
		f << "beta0_sigma = " << beta0_sigma << "\t; sigma\n";
		break;
	case 0:
	default:
		f << "beta0 = " << beta0 << "\t; default scent secretion\n";
	}
	// searcher chemodeposition relaxation rate
	f << "betaDecayTime = " << betaDecayTime << "\t; scent secretion rate relaxation time\n";
	// rotational chemosensitivity
	switch (chiRot_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
		f << "chiRot_distr = bimodal\t; rotational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiRot_min = " << chiRot_min << "\t; min\n";
		f << "chiRot_max = " << chiRot_max << "\t; max\n";
		f << "chiRot_bias = " << chiRot_bias << "\t; bias\n";
		break;
	case 2:
		f << "chiRot_distr = uniform\t; rotational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiRot_min = " << chiRot_min << "\t; min\n";
		f << "chiRot_max = " << chiRot_max << "\t; max\n";
		f << "chiRot_bias = " << chiRot_bias << "\t; bias\n";
		break;
	case 3:
		f << "chiRot_distr = log-uniform\t; rotational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiRot_min = " << chiRot_min << "\t; min\n";
		f << "chiRot_max = " << chiRot_max << "\t; max\n";
		f << "chiRot_bias = " << chiRot_bias << "\t; bias\n";
		break;
	case 4:
		f << "chiRot_distr = gaussian\t; rotational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiRot_min = " << chiRot_min << "\t; min\n";
		f << "chiRot_max = " << chiRot_max << "\t; max\n";
		f << "chiRot_mean = " << chiRot_mean << "\t; mean\n";
		f << "chiRot_sigma = " << chiRot_sigma << "\t; sigma\n";
		break;
	case 5:
		f << "chiRot_distr = log-gaussian\t; rotational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiRot_min = " << chiRot_min << "\t; min\n";
		f << "chiRot_max = " << chiRot_max << "\t; max\n";
		f << "chiRot_mean = " << chiRot_mean << "\t; mean\n";
		f << "chiRot_sigma = " << chiRot_sigma << "\t; sigma\n";
		break;
	case 0:
	default:
		f << "chiRot = " << chiRot << "\t; rotational chemosensitivity\n";
	}
	// translational chemosensitivity
	switch (chiTrans_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
		f << "chiTrans_distr = bimodal\t; translational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiTrans_min = " << chiTrans_min << "\t; min\n";
		f << "chiTrans_max = " << chiTrans_max << "\t; max\n";
		f << "chiTrans_bias = " << chiTrans_bias << "\t; bias\n";
		break;
	case 2:
		f << "chiTrans_distr = uniform\t; translational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiTrans_min = " << chiTrans_min << "\t; min\n";
		f << "chiTrans_max = " << chiTrans_max << "\t; max\n";
		f << "chiTrans_bias = " << chiTrans_bias << "\t; bias\n";
		break;
	case 3:
		f << "chiTrans_distr = log-uniform\t; translational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiTrans_min = " << chiTrans_min << "\t; min\n";
		f << "chiTrans_max = " << chiTrans_max << "\t; max\n";
		f << "chiTrans_bias = " << chiTrans_bias << "\t; bias\n";
		break;
	case 4:
		f << "chiTrans_distr = gaussian\t; translational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiTrans_min = " << chiTrans_min << "\t; min\n";
		f << "chiTrans_max = " << chiTrans_max << "\t; max\n";
		f << "chiTrans_mean = " << chiTrans_mean << "\t; mean\n";
		f << "chiTrans_sigma = " << chiTrans_sigma << "\t; sigma\n";
		break;
	case 5:
		f << "chiTrans_distr = log-gaussian\t; translational chemosensitivity: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "chiTrans_min = " << chiTrans_min << "\t; min\n";
		f << "chiTrans_max = " << chiTrans_max << "\t; max\n";
		f << "chiTrans_mean = " << chiTrans_mean << "\t; mean\n";
		f << "chiTrans_sigma = " << chiTrans_sigma << "\t; sigma\n";
		break;
	case 0:
	default:
		f << "chiTrans = " << chiTrans << "\t; translational chemosensitivity\n";
	}
	// scent noise level
	switch (SC0_distr)
	{ // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
	case 1:
		f << "SC0_distr = bimodal\t; scent noise level: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "SC0_min = " << SC0_min << "\t; min\n";
		f << "SC0_max = " << SC0_max << "\t; max\n";
		f << "SC0_bias = " << SC0_bias << "\t; bias\n";
		break;
	case 2:
		f << "SC0_distr = uniform\t; scent noise level: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "SC0_min = " << SC0_min << "\t; min\n";
		f << "SC0_max = " << SC0_max << "\t; max\n";
		f << "SC0_bias = " << SC0_bias << "\t; bias\n";
		break;
	case 3:
		f << "SC0_distr = log-uniform\t; scent noise level: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "SC0_min = " << SC0_min << "\t; min\n";
		f << "SC0_max = " << SC0_max << "\t; max\n";
		f << "SC0_bias = " << SC0_bias << "\t; bias\n";
		break;
	case 4:
		f << "SC0_distr = gaussian\t; scent noise level: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "SC0_min = " << SC0_min << "\t; min\n";
		f << "SC0_max = " << SC0_max << "\t; max\n";
		f << "SC0_mean = " << SC0_mean << "\t; mean\n";
		f << "SC0_sigma = " << SC0_sigma << "\t; sigma\n";
		break;
	case 5:
		f << "SC0_distr = log-gaussian\t; scent noise level: single [value], bimodal, uniform, log-uniform, gaussian, log-gaussian\n";
		f << "SC0_min = " << SC0_min << "\t; min\n";
		f << "SC0_max = " << SC0_max << "\t; max\n";
		f << "SC0_mean = " << SC0_mean << "\t; mean\n";
		f << "SC0_sigma = " << SC0_sigma << "\t; sigma\n";
		break;
	case 0:
	default:
		f << "SC0 = " << SC0 << "\t; scent noise level\n";
	}
	f << "\n";
	f << "[Pairwise]\n";
	f << "PPepsilon = " << PPepsilon << "\t; soft potential epsilon\n";
	f << "PPsigma = " << PPsigma << "\t; soft potential sigma\n";
	f << "\n";
	f << "[System]\n";
	f << "boxSize = " << boxSize << "\t; the grid on which the concentration shall be calculated is 2L-by-2L\n";
	f << "boundaryType = " << boundaryType << "\t; boundary: 0 - circle, 1 - square, 2 - square PBC\n";
	f << "scentDecayTime = " << scentDecayTime << "\t; scent decay time\n";
	f << "initialNTRtype = " << initialNTRtype << "\t; initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)\n";
	f << "initialParticlePos = " << initialParticlePos << "\t; 0 - center of arena, 1 - uniform distribution across arena area\n";
	f << "\n";
	f << "[Home]\n";
	f << "homeType = " << homeType << "\t; home type: 1 - parabolic potential, 2 - conical potential\n";
	f << "homeRadius = " << homeRadius << "\t; radius of the circular area at the origin called 'home'\n";
	f << "homePotentialKT = " << homePotentialKT << "\t; strength of the home potential, translational strength\n";
	f << "homePotentialKR = " << homePotentialKR << "\t; strength of the home potential, rotational strength\n";
	f << "\n";
	f << "[TimedReset]\n";
	f << "timedResetTimerType = " << timedResetTimerType << "\t; timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution\n";
	f << "timedResetType = " << timedResetType << "\t; timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward\n";
	f << "timedResetMeanTime = " << timedResetMeanTime << "\t; average time between timed resets\n";
	f << "timedResetHomePotential = " << timedResetHomePotential << "\t; does timed reset switches on home potential sensing\n";
	f << "globalResetTime = " << globalResetTime << "\t; time marker for all-particle reset, default -1 (not set), active when non-negative\n";
	f << "\n";
	f << "[Boundary]\n";
	f << "boundaryResetType = " << boundaryResetType << "\t; boundary reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward\n";
	f << "betaBoundaryMult = " << betaBoundaryMult << "\t; beta multiplication coefficient when particle hits boundary\n";
	f << "VBoundary = " << VBoundary << "\t; velocity when particle hits boundary\n";
	f << "boundaryResetSRtime = " << boundaryResetSRtime << "\t; is boundary hit resets stochastic reset time or not\n";
	f << "boundaryHomePotential = " << boundaryHomePotential << "\t; does boundary hit switches on home potential sensing\n";
	f << "\n";
	f << "[TargetList]\n";
	f << "targetNtargets = " << targetList.size() << "\t; number of targets in the list\n";
	f << "\n";
	for (int nt = 0; nt < targetList.size(); nt++)
	{
		std::string tname = "Target" + vc3_general::itoa(nt + 1);
		f << "[" << tname << "]\n";
		f << "targetDistanceMin = " << targetList[nt].targetDistanceMin << "\t; minimum distance from the origin\n";
		f << "targetDistanceMax = " << targetList[nt].targetDistanceMax << "\t; maximum distance from the origin\n";
		f << "targetAngleMin = " << targetList[nt].targetAngleMin << "\t; minimum anglular position, in degrees\n";
		f << "targetAngleMax = " << targetList[nt].targetAngleMax << "\t; maximum anglular position, in degrees\n";
		f << "targetRadiusMin = " << targetList[nt].targetRadiusMin << "\t; minimum target radius\n";
		f << "targetRadiusMax = " << targetList[nt].targetRadiusMax << "\t; maximum target radius\n";
		f << "targetWeightMin = " << targetList[nt].targetWeightMin << "\t; minimum target weight (number of first hits to capture)\n";
		f << "targetWeightMax = " << targetList[nt].targetWeightMax << "\t; maximum target weight (number of first hits to capture)\n";
		f << "targetAppearType = " << targetList[nt].targetAppearType << "\t; 0 - appear by time, 1 - appear after termination\n";
		f << "targetAppearTime = " << targetList[nt].targetAppearTime << "\t; loop appear time\n";
		f << "targetAppearDelay = " << targetList[nt].targetAppearDelay << "\t; time delay before appear trigger and real appear of the target\n";
		f << "targetTerminateType = " << targetList[nt].targetTerminateType << "\t; 0 - terminate by time, 1 - terminate after capture\n";
		f << "targetTerminateTime = " << targetList[nt].targetTerminateTime << "\t; loop terminate time\n";
		f << "targetResetType = " << targetList[nt].targetResetType << "\t; target reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward\n";
		f << "betaTargetMultMin = " << targetList[nt].betaTargetMultMin << "\t; minimum beta multiplication coefficient when particle hits target\n";
		f << "betaTargetMultMax = " << targetList[nt].betaTargetMultMax << "\t; maximum beta multiplication coefficient when particle hits target\n";
		f << "VTargetMin = " << targetList[nt].VTargetMin << "\t; minimum particle velocity set coefficient when particle hits target\n";
		f << "VTargetMax = " << targetList[nt].VTargetMax << "\t; maximum particle velocity set coefficient when particle hits target\n";
		f << "targetResetSRtime = " << targetList[nt].targetResetSRtime << "\t; is target hit resets stochastic reset time or not\n";
		f << "targetHomePotential = " << targetList[nt].targetHomePotential << "\t; does target hit switches on home potential sensing\n";
		f << "\n";
	}
	f << "[Computations]\n";
	f << "latticeSize = " << latticeSize << "\t; number of nodes on each side of the grid\n";
	f << "timeStep = " << timeStep << "\t; length of time step\n";
	f << "Rscent_cutoffMultiplier = " << Rscent_cutoffMultiplier << "\t; scent radius cutoff multiplier\n";
	f << "scentDecayRescalingNsteps = " << scentDecayRescalingNsteps << "\t; number of steps between inflation rescaling\n";
	f << "gpuHistoryLength = " << gpuHistoryLength << "\t; number of steps stored on GPU before it's flushed to CPU and analyzed\n";
	f << "\n";
	f << "[Run]\n";
	f << "nSteps = " << nSteps << "\t; total number of time steps in each iteration\n";
	f << "nIterations = " << nIterations << "\t; total number of iterations\n";
	f << "\n";
	f << "[Scenarios]\n";
	f << "scenarioParticlesNscenarios = " << scenarioParticlesList.size() << "\t; number of scenarios for particle property changes\n";
	f << "\n";
	for (int ns = 0; ns < scenarioParticlesList.size(); ns++)
	{
		std::string sname = "scenarioParticles" + vc3_general::itoa(ns + 1);
		f << "[" << sname << "]\n";
		if(scenarioParticlesList[ns].parameter == 1) f << "scenarioParticlesParameter = particle_V0\t; \n";
		else if(scenarioParticlesList[ns].parameter == 2) f << "scenarioParticlesParameter = particle_beta0\t; \n";
		else if(scenarioParticlesList[ns].parameter == 3) f << "scenarioParticlesParameter = particle_chiT\t; \n";
		else if(scenarioParticlesList[ns].parameter == 4) f << "scenarioParticlesParameter = particle_chiR\t; \n";
		else if(scenarioParticlesList[ns].parameter == 5) f << "scenarioParticlesParameter = particle_c0\t; \n";
		else if(scenarioParticlesList[ns].parameter == 6) f << "scenarioParticlesParameter = particle_DR\t; \n";
		else f << "scenarioParticlesParameter = unknown\t; CHECK THIS ERROR\n";
		f << "scenarioParticlesTimesteppoints = ";
		for (int q = 0; q < scenarioParticlesList[ns].timesteppoints.size(); q++)
			f << scenarioParticlesList[ns].timesteppoints[q] << " ";
		f << "\t; \n";
		f << "scenarioParticlesValues = ";
		for (int q = 0; q < scenarioParticlesList[ns].values.size(); q++)
			f << scenarioParticlesList[ns].values[q] << " ";
		f << "\t; \n";
	}
	f << "\n";
	f << "[Output]\n";
	f << "cumulativeDataMinStep = " << cumulativeDataMinStep << "\t; minimum step after which all cumulative data to be collected\n";
	f << "cumulativePCblockSize = " << cumulativePCblockSize << "\t; block size for the cumulative particle concentration\n";
	f << "cumulativeRnbins = " << cumulativeRnbins << "\t; number of bins in cumulative radial particle distribution\n";
	f << "PCReg_startstep = " << PCReg_startstep << "\t; \n";
	f << "PCReg_window = " << PCReg_window << "\t; \n";
	f << "PCReg_blockSize = " << PCReg_blockSize << "\t; \n";
	f << "SCReg_startstep = " << SCReg_startstep << "\t; \n";
	f << "SCReg_window = " << SCReg_window << "\t; \n";
	f << "SCReg_blockSize = " << SCReg_blockSize << "\t; \n";
	/*f << "stepDrHistogram_min = " << stepDrHistogram_min << "\t; limits of step dr histogram\n";
	f << "stepDrHistogram_max = " << stepDrHistogram_max << "\t; limits of step dr histogram\n";
	f << "stepDrHistogram_nbins = " << stepDrHistogram_nbins << "\t; number of bins in dtep dr histogram\n";
	f << "totalScentReg_window = " << totalScentReg_window << "\t; regular time total scent will be averaged inside window in each iteration, in steps\n";
	f << "totalScentReg_avgperiod = " << totalScentReg_avgperiod << "\t; regular time total scent will be averaged with this period in each iteration, in steps\n";
	f << "PCrtheta_r_nbins = " << PCrtheta_r_nbins << "\t; number of bins in rtheta PC hist over radius\n";
	f << "PCrtheta_theta_binsize = " << PCrtheta_theta_binsize << "\t; linear size of bins in rtheta PC hist over theta\n";
	f << "PCrthetaReg_window = " << PCrthetaReg_window << "\t; regular time rtheta PC will be accumulated inside window in each iteration, in steps\n";
	f << "PCtheta_nbins = " << PCtheta_nbins << "\t; number of bins in theta PC hist\n";
	f << "PCtheta_rmin = " << PCtheta_rmin << "\t; r threshold after which particles participates in theta PC hist\n";
	f << "PCtheta_rmax = " << PCtheta_rmax << "\t; r threshold after which particles participates in theta PC hist\n";
	f << "PCthetaReg_window = " << PCthetaReg_window << "\t; regular time theta PC will be accumulated inside window in each iteration, in steps\n";*/
	f << "HitCountReg_window = " << HitCountReg_window << "\t; regular time interval at which hit counts are stored, in steps\n";
	f << "HitCountReg_minStep = " << HitCountReg_minStep << "\t; minimum step after which all hit counts are on, in steps \n";
	f << "HitCountPP_afterReset = " << HitCountPP_afterReset << "\t; does per-particle hit count counts only after the first reset\n";
	f << "printTrjs = " << printTrjs << "\t; print trajectories flag\n";
	f << "trjStepFrom = " << trjStepFrom << "\t; print trajectories from this step\n";
	f << "trjStepTo = " << trjStepTo << "\t; print trajectories to this step (negative = until the end of sim)\n";
	f << "trjBlockEverySteps = " << trjBlockEverySteps << "\t; \n";
	f << "trjBlockDurationSteps = " << trjBlockDurationSteps << "\t; \n";
	f << "trjBlockFreqSteps = " << trjBlockFreqSteps << "\t; \n";
	f << "trjFormat = " << trjFormat << "\t; trajectories format: 0 - LAMMPS x y vx vy\n";
	f << "trjParticleProperties = " << trjParticleProperties << "\t; trajectory particle properties\n";
	f << "trjPrecision = " << trjPrecision << "\t; precision of numerical data in trj output: 0 - default (6 digits), 1 - extended (10 digits), 2 - full\n";
	/*f << "computeMSD = " << computeMSD << "\t; Compute MSD of the trajectories\n";
	f << "maxMSDLength = " << maxMSDLength << "\t; length of MSD output range\n";
	f << "MSDstep = " << MSDstep << "\t; step count between the nearest MSD points\n";
	f << "PRH_min = " << PRH_min << "\t; limits of path-rotation histogram\n";
	f << "PRH_max = " << PRH_max << "\t; limits of path-rotation histogram\n";
	f << "PRH_nbins = " << PRH_nbins << "\t; number of bins in path-rotation histogram\n";
	f << "PRH_step = " << PRH_step << "\t; step between the nearest points for PRH\n";
	f << "GTFPT_blockSize = " << GTFPT_blockSize << "\t; block size for GT FPT collection\n";
	f << "GTFPT_Rtarget = ";
	for (int q = 0; q < GTFPT_Rtarget.size(); q++)
	{
		if (q > 0) f << ", ";
		f << GTFPT_Rtarget[q];
	}
	f << "\t; GT FPT target radius vector\n";
	f << "\n";*/

	f.close();
	std::cout << "\nParameter file \"" << filename << "\" written successfully\n";
	return 0;
}


#endif
