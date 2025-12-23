#ifndef VC3_PHYS_KSNP_KERNELS_MOVEPARTICLES
#define VC3_PHYS_KSNP_KERNELS_MOVEPARTICLES


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cubasicmath.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"



/** Device kernel
    Produce set of procedures to move particles across the arena

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_moveParticles(vc3_phys::gpu_variables* gpuvariables, 
    vc3_phys::gpu_targets* gputargets, const vc3_phys::gpu_matrixes* __restrict__ gpumatrixes, 
    vc3_phys::gpu_particles* gpuparticles)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;
    
    // $ calculateGradientBicubic()
    // ============================
    
    // Gradient-inconsistent bicubic interpolation
    // 1. Calculate current position
    vc3_cumath::planar::cuvector cpos(gpuparticles->particle_pos[poffset].x - gpuparticles->particle_posLattice[poffset].x * __gpuprecomputes.dl,
                                    gpuparticles->particle_pos[poffset].y - gpuparticles->particle_posLattice[poffset].y * __gpuprecomputes.dl);
    vc3_cumath::planar::cuvector regpos = cpos * __gpuprecomputes.invdl; /*! CAN BE OPTIMIZED - NO cpos NEEDED !*/
    vc3_cumath::planar::cuvector regpos2(regpos.x * regpos.x, regpos.y * regpos.y);

    // 2. Calculate a(y), b(y), c(y) and d(y) for the four y points
    // ! NOTE: these calculations are hard-binded to the SC offset style!
    flt2 ay[4], by[4], cy[4], dy[4];
    long long int offsetm1m1 = vc3_phys::get_SCoffset(gpuparticles->particle_posLattice[poffset].x - 1, gpuparticles->particle_posLattice[poffset].y - 1, __gpusetup_variables.latticeSize);
    long long int offset0m1 =  vc3_phys::get_SCoffset(gpuparticles->particle_posLattice[poffset].x    , gpuparticles->particle_posLattice[poffset].y - 1, __gpusetup_variables.latticeSize);
    long long int offset1m1 =  vc3_phys::get_SCoffset(gpuparticles->particle_posLattice[poffset].x + 1, gpuparticles->particle_posLattice[poffset].y - 1, __gpusetup_variables.latticeSize);
    long long int offset2m1 =  vc3_phys::get_SCoffset(gpuparticles->particle_posLattice[poffset].x + 2, gpuparticles->particle_posLattice[poffset].y - 1, __gpusetup_variables.latticeSize);
    for (int y = 0; y < 4; y++)
    {
        ay[y] = gpumatrixes->SC[offset0m1 + y];
        cy[y] = (gpumatrixes->SC[offset1m1 + y] + gpumatrixes->SC[offsetm1m1 + y] - 2.00 * gpumatrixes->SC[offset0m1 + y]) * 0.50;
        dy[y] = (3.00 * (gpumatrixes->SC[offset0m1 + y] - gpumatrixes->SC[offset1m1 + y])
            + gpumatrixes->SC[offset2m1 + y] - gpumatrixes->SC[offsetm1m1 + y]) / 6.00;
        by[y] = gpumatrixes->SC[offset1m1 + y] - ay[y] - cy[y] - dy[y];
    }

    // 3. Calculate alpha, beta, gamma, delta for a, b, c, d
    flt2 alpha[4], beta[4], gamma[4], delta[4];
    alpha[0] = ay[1];
    gamma[0] = (ay[2] + ay[0] - 2.00 * ay[1]) * 0.50;
    delta[0] = (3.00 * (ay[1] - ay[2]) + ay[3] - ay[0]) / 6.00;
    beta[0] = ay[2] - alpha[0] - gamma[0] - delta[0];
    alpha[1] = by[1];
    gamma[1] = (by[2] + by[0] - 2.00 * by[1]) * 0.50;
    delta[1] = (3.00 * (by[1] - by[2]) + by[3] - by[0]) / 6.00;
    beta[1] = by[2] - alpha[1] - gamma[1] - delta[1];
    alpha[2] = cy[1];
    gamma[2] = (cy[2] + cy[0] - 2.00 * cy[1]) * 0.50;
    delta[2] = (3.00 * (cy[1] - cy[2]) + cy[3] - cy[0]) / 6.00;
    beta[2] = cy[2] - alpha[2] - gamma[2] - delta[2];
    alpha[3] = dy[1];
    gamma[3] = (dy[2] + dy[0] - 2.00 * dy[1]) * 0.50;
    delta[3] = (3.00 * (dy[1] - dy[2]) + dy[3] - dy[0]) / 6.00;
    beta[3] = dy[2] - alpha[3] - gamma[3] - delta[3];

    // 4. Calculated y-adjusted a, b, c, d and gradients
    flt2 a = alpha[0] + beta[0] * regpos.y + gamma[0] * regpos2.y + delta[0] * regpos2.y * regpos.y;
    flt2 b = alpha[1] + beta[1] * regpos.y + gamma[1] * regpos2.y + delta[1] * regpos2.y * regpos.y;
    flt2 c = alpha[2] + beta[2] * regpos.y + gamma[2] * regpos2.y + delta[2] * regpos2.y * regpos.y;
    flt2 d = alpha[3] + beta[3] * regpos.y + gamma[3] * regpos2.y + delta[3] * regpos2.y * regpos.y;
    flt2 dady = beta[0] + 2.00 * gamma[0] * regpos.y + 3.00 * delta[0] * regpos2.y;
    flt2 dbdy = beta[1] + 2.00 * gamma[1] * regpos.y + 3.00 * delta[1] * regpos2.y;
    flt2 dcdy = beta[2] + 2.00 * gamma[2] * regpos.y + 3.00 * delta[2] * regpos2.y;
    flt2 dddy = beta[3] + 2.00 * gamma[3] * regpos.y + 3.00 * delta[3] * regpos2.y;

    // 5. Calculate interpolated function and gradient, with scent inflation correction
    flt2 ic = 1.00 / gpuvariables->scentDivideFactor;
    gpuparticles->particle_SC[poffset] = (a + b * regpos.x + c * regpos2.x + d * regpos2.x * regpos.x) * ic;
    gpuparticles->particle_GSC[poffset].x = (b + 2.00 * c * regpos.x + 3.00 * d * regpos2.x) * __gpuprecomputes.invdl * ic;
    gpuparticles->particle_GSC[poffset].y = (dady + dbdy * regpos.x + dcdy * regpos2.x + dddy * regpos2.x * regpos.x) * __gpuprecomputes.invdl * ic;


    // $ updatePosition()
    // ==================
    // 1. Reset hit flags
    gpuparticles->particle_flag_boundaryHit[poffset] = false;
    gpuparticles->particle_flag_targetHit[poffset] = -1;
    gpuparticles->particle_flag_timedReset[poffset] = false;
    //_homeHit = false;
    
    // 2. Velocity
    vc3_cumath::planar::cuvector v;
    if (__gpusetup_variables.chemosensitivityModel == 1) // gradient of log(SC)
    {
        v = gpuparticles->particle_chiT[poffset] * gpuparticles->particle_GSC[poffset] / (gpuparticles->particle_SC[poffset] + __gpusetup_variables.SC0)
            + gpuparticles->particle_PPforceSpatialHashing[poffset]
            + __gpusetup_variables.V0 * gpuparticles->particle_rot[poffset];
    }
    else if (__gpusetup_variables.chemosensitivityModel == 0) // gradient of SC
    {
        v = gpuparticles->particle_chiT[poffset] * gpuparticles->particle_GSC[poffset] 
            + gpuparticles->particle_PPforceSpatialHashing[poffset]
            + __gpusetup_variables.V0 * gpuparticles->particle_rot[poffset];
    }
    if (__gpusetup_variables.keepV0Constant)
    {
        v.normalize(__gpusetup_variables.V0);
    }

    // 3. Home sensing potential
    vc3_cumath::planar::cuvector u(0.00, 0.00);/*, opos(_pos);
    if (_senseHome && _param.homeType > 0)
    { // 0 - instant reset to the origin
        opos.x -= _param.boxSize * 0.50;
        opos.y -= _param.boxSize * 0.50;
        if (_param.homeType == 1)
        { // 1 - parabolic potential
            u = -2.00 * _param.homePotentialKT * opos;
        }
        else if (_param.homeType == 2)
        { // 2 - conical potential
            u = -2.00 * _param.homePotentialKT / opos.length() * opos;
        }
    }*/

    // 4. Total motion
    // Detect errors
    if (isnan(v.x) || isnan(v.y) || isnan(u.x) || isnan(u.y))
    {
        //flt2 sc0 = gpumatrixes->SC[offset0m1 + 0], sc1 = gpumatrixes->SC[offset0m1 + 1], sc2 = gpumatrixes->SC[offset0m1 + 2], sc3 = gpumatrixes->SC[offset0m1 + 3];
        atomicAdd(&(gpuvariables->error_vnan), 1);
        printf("ERR VNAN: step=%lld, time=%.15le, pID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset);
    }
    gpuparticles->particle_dpos[poffset] = __gpusetup_variables.timeStep * (v + u);
    gpuparticles->particle_pos[poffset] += gpuparticles->particle_dpos[poffset];
    gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
    gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
    if (isnan(gpuparticles->particle_pos[poffset].x) || isnan(gpuparticles->particle_pos[poffset].y))
    {
        atomicAdd(&(gpuvariables->error_posnan), 1);
        printf("ERR POSNAN: step=%lld, time=%.15le, pID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset);
    }

    // $ updateRotation()
    // ==================
    // 1. Linear scheme
    vc3_cumath::planar::cuvector gp(-gpuparticles->particle_GSC[poffset].y, gpuparticles->particle_GSC[poffset].x);
    flt2 absg = gp.length(), sintheta = vc3_cumath::planar::SP(gpuparticles->particle_rot[poffset], gp), 
        gcostheta = vc3_cumath::planar::SP(gpuparticles->particle_rot[poffset], gpuparticles->particle_GSC[poffset]);
    if (absg > 1.0e-15) sintheta /= absg;
    else sintheta = 0.00;
    flt2 omega;
    if (__gpusetup_variables.chemosensitivityModel == 1) // gradient of log(SC)
    {
        omega = gpuparticles->particle_chiR[poffset] * gcostheta * sintheta / (gpuparticles->particle_SC[poffset] + __gpusetup_variables.SC0);
    }
    else if (__gpusetup_variables.chemosensitivityModel == 0) // gradient of SC
    {
        omega = gpuparticles->particle_chiR[poffset] * gcostheta * sintheta;
    }
    // 2. Home sensing potential
    flt2 uomega = 0.00;
    /*if (_senseHome && _param.homeType > 0)
    { // 0 - instant reset to the origin
        v3_math::planar::vector<double> odir(_param.boxSize * 0.50 - _pos.x, _param.boxSize * 0.50 - _pos.y);
        double absd = odir.length();
        double cp = (_rot.x * odir.y - _rot.y * odir.x);
        if (absd > 1.0e-8) cp /= absd;
        else cp = 0.00;
        if (_param.homeType == 1)
        { // 1 - parabolic potential
            uomega = _param.homePotentialKR * cp * absd;
        }
        else if (_param.homeType == 2)
        { // 2 - conical potential
            uomega = _param.homePotentialKR * cp;
        }
    }*/
    if (isnan(omega) || isnan(uomega))
    {
        atomicAdd(&(gpuvariables->error_omeganan), 1);
        printf("ERR OMEGANAN: step=%lld, time=%.15le, pID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset);
    }
    flt2 z1 = curand_normal_double(&(gpuparticles->particle_curandstate[poffset]));
    flt2 dphi = __gpusetup_variables.timeStep * (omega + uomega) + vc3_cumath::msqrt_flt2(2.00 * __gpusetup_variables.timeStep * __gpusetup_variables.rotationalDiffusion) * z1;
    gpuparticles->particle_angle[poffset] += dphi;
    sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));

    // $ targetCheck()
    // ===============
    // 1. Check target hit
    flt2 rt2hit = 0.00;
    int nthit = -1;
    for (int nt = 0; nt < __gpusetup_variables.nTargets; nt++)
    {
        if (gputargets->target_active[nt])
        {
            flt2 rt2 = (gpuparticles->particle_pos[poffset] - gputargets->target_pos[nt]).length2();
            /*printf("rt2: step=%lld, time=%.15le\ttID=%d\t%.8le\t%.8le\t%.8le\t%.8le\t%.8le\n", gpuvariables->currentStep, gpuvariables->currentTime, nt,
                gpuparticles->particle_pos[poffset].x, gpuparticles->particle_pos[poffset].y, 
                gputargets->target_pos[nt].x, gputargets->target_pos[nt].y, rt2);*/
            if (rt2 <= gputargets->target_radius2[nt])
            {
                /*printf("TH: step=%lld, time=%.15le\ttID=%d\t%.8le\t%.8le\t%.8le\t%.8le\t%.8le\t%.8le\n", gpuvariables->currentStep, gpuvariables->currentTime, nt,
                    gpuparticles->particle_pos[poffset].x, gpuparticles->particle_pos[poffset].y,
                    gputargets->target_pos[nt].x, gputargets->target_pos[nt].y, gputargets->target_radius2[nt], rt2);*/
                if (gpuparticles->particle_flag_targetHit[poffset] < 0)
                { // First hit detected
                    nthit = nt;
                    rt2hit = rt2;
                }
                else if (rt2 < rt2hit)
                { // Choose closest
                    nthit = nt;
                    rt2hit = rt2;
                }
            }
        }
    }
    gpuparticles->particle_flag_targetHit[poffset] = nthit;
    // 2. React to target hit
    if (nthit >= 0)
    {
        //printf("nthit>=0: step=%lld, time=%.15le, pID=%d, tID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset, nthit);
        // Update target hit counter
        atomicAdd(&(gputargets->target_hit_count[nthit]), 1);

        // Update particle hitting the target
        if (gputargets->target_resetType[nthit] == 0)
        { // 0 - particle position to the center + random direction
            //resetToTheOrigin();
            gpuparticles->particle_pos[poffset].x = __gpuprecomputes.halfBox;
            gpuparticles->particle_pos[poffset].y = __gpuprecomputes.halfBox;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
            flt2 z2 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
            gpuparticles->particle_angle[poffset] = vc3_cumath::TwoPi * z2;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
        }
        else if (gputargets->target_resetType[nthit] == 1)
        { // 1 - direction to the center
            vc3_cumath::planar::cuvector opos(gpuparticles->particle_pos[poffset].x - __gpuprecomputes.halfBox,
                gpuparticles->particle_pos[poffset].y - __gpuprecomputes.halfBox);
            gpuparticles->particle_angle[poffset] = opos.phi() + vc3_cumath::Pi;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
        }
        else if (gputargets->target_resetType[nthit] == 2)
        { // 2 - reverse direction + revert position one timestep backward
            gpuparticles->particle_angle[poffset] += vc3_cumath::Pi;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
            gpuparticles->particle_pos[poffset] += (-1.00) * gpuparticles->particle_dpos[poffset];
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
        }
    }


    // $ boundaryCheck()
    // =================
    vc3_cumath::planar::cuvector opos(gpuparticles->particle_pos[poffset].x - __gpuprecomputes.halfBox,
                                    gpuparticles->particle_pos[poffset].y - __gpuprecomputes.halfBox);
    // 1. Check boundary hit
    if (__gpusetup_variables.boundaryType == 1) // Circular arena
    {
        if (opos.length2() >= (__gpuprecomputes.halfBox - __gpuprecomputes.dl) * (__gpuprecomputes.halfBox - __gpuprecomputes.dl))
            gpuparticles->particle_flag_boundaryHit[poffset] = true;
    }
    else if (__gpusetup_variables.boundaryType == 0) // Square arena
    {
        if (gpuparticles->particle_pos[poffset].x <= __gpuprecomputes.dl ||
            gpuparticles->particle_pos[poffset].x >= __gpusetup_variables.boxSize - __gpuprecomputes.dl ||
            gpuparticles->particle_pos[poffset].y <= __gpuprecomputes.dl ||
            gpuparticles->particle_pos[poffset].y >= __gpusetup_variables.boxSize - __gpuprecomputes.dl)
            gpuparticles->particle_flag_boundaryHit[poffset] = true;
    }
    // 2. React to boundary hit
    if (gpuparticles->particle_flag_boundaryHit[poffset])
    {
        if (__gpusetup_variables.boundaryResetType == 0) 
        { // position to the center + random direction
            //resetToTheOrigin();
            gpuparticles->particle_pos[poffset].x = __gpuprecomputes.halfBox;
            gpuparticles->particle_pos[poffset].y = __gpuprecomputes.halfBox;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
            flt2 z2 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
            gpuparticles->particle_angle[poffset] = vc3_cumath::TwoPi * z2;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
        }
        else if (__gpusetup_variables.boundaryResetType == 1) 
        { // direction to the center + one timestep towards center
            gpuparticles->particle_angle[poffset] = opos.phi() + vc3_cumath::Pi;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
            vc3_cumath::planar::cuvector dcpos = __gpusetup_variables.timeStep * __gpusetup_variables.V0 * gpuparticles->particle_rot[poffset];
            gpuparticles->particle_dpos[poffset] += dcpos;
            gpuparticles->particle_pos[poffset] += dcpos;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
        }
        else if (__gpusetup_variables.boundaryResetType == 2)
        { // 2 - reverse direction + one timestep backward
            gpuparticles->particle_angle[poffset] += vc3_cumath::Pi;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
            gpuparticles->particle_pos[poffset] += (-1.00) * gpuparticles->particle_dpos[poffset];
            gpuparticles->particle_dpos[poffset].x = gpuparticles->particle_dpos[poffset].y = 0.00;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
            /*! Be warned about possible errorous consequent boundary hits !*/
        }
        if (isnan(gpuparticles->particle_pos[poffset].x) || isnan(gpuparticles->particle_pos[poffset].y))
        {
            atomicAdd(&(gpuvariables->error_bhposnan), 1);
            printf("ERR BHPOSNAN: step=%lld, time=%.15le, pID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset);
        }
        /*! CHECK FOR STILL-OUT-OF-THE-BOUNDARY ERROR !*/
    }

    // $ timedReset()
    // ==============
    // 1. Check boundary hit conditions
    bool NRTupdated = false;
    if (gpuparticles->particle_flag_boundaryHit[poffset] && __gpusetup_variables.boundaryResetSRtime)
    {
        if (__gpusetup_variables.timedResetTimerType == 0)
        {
            gpuparticles->particle_NRT[poffset] += __gpusetup_variables.timedResetMeanTime;
            NRTupdated = true;
        }
        else if (__gpusetup_variables.timedResetTimerType == 1)
        {
            flt2 z3 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
            gpuparticles->particle_NRT[poffset] = gpuvariables->currentTime - log(z3) * __gpusetup_variables.timedResetMeanTime;
            NRTupdated = true;
        }
    }

    // 2. Check target hit conditions
    if (gpuparticles->particle_flag_targetHit[poffset] >= 0 && !NRTupdated)
    {
        if (gputargets->target_resetSRtime[gpuparticles->particle_flag_targetHit[poffset]])
        {
            if (__gpusetup_variables.timedResetTimerType == 0)
            {
                gpuparticles->particle_NRT[poffset] += __gpusetup_variables.timedResetMeanTime;
                NRTupdated = true;
            }
            else if (__gpusetup_variables.timedResetTimerType == 1)
            {
                flt2 z3 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
                gpuparticles->particle_NRT[poffset] = gpuvariables->currentTime - log(z3) * __gpusetup_variables.timedResetMeanTime;
                NRTupdated = true;
            }
        }
    }

    // 3. Check stochastic resetting conditions
    if (gpuvariables->currentTime >= gpuparticles->particle_NRT[poffset] && __gpusetup_variables.timedResetMeanTime > 0.00)
    {
        // Set stochastic reset flag
        gpuparticles->particle_flag_timedReset[poffset] = true;

        // Produce stochastic reset
        if (__gpusetup_variables.timedResetType == 0)
        { // 0 - position to the center + random direction
            //resetToTheOrigin();
            gpuparticles->particle_pos[poffset].x = __gpuprecomputes.halfBox;
            gpuparticles->particle_pos[poffset].y = __gpuprecomputes.halfBox;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
            flt2 z2 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
            gpuparticles->particle_angle[poffset] = vc3_cumath::TwoPi * z2;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
            /* Left empty until further times when anything else works and scales good
            _targetHitN = 0;*/
        }
        /* Left empty until further times when anything else works and scales good
        else if (__gpusetup_variables.timedResetType == 1 && !(_targetHit || _boundaryHit)) */
        else if (__gpusetup_variables.timedResetType == 1 && !(gpuparticles->particle_flag_boundaryHit[poffset]))
        { // 1 - direction to the center + one timestep towards center
            gpuparticles->particle_angle[poffset] = opos.phi() + vc3_cumath::Pi;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
            vc3_cumath::planar::cuvector dcpos = __gpusetup_variables.timeStep * __gpusetup_variables.V0 * gpuparticles->particle_rot[poffset];
            gpuparticles->particle_dpos[poffset] += dcpos;
            gpuparticles->particle_pos[poffset] += dcpos;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
        }
        /* Left empty until further times when anything else works and scales good
        else if (__gpusetup_variables.timedResetType == 2 && !(_targetHit || _boundaryHit)) */
        else if (__gpusetup_variables.timedResetType == 2 && !(gpuparticles->particle_flag_boundaryHit[poffset]))
        { // 2 - reverse direction + one timestep backward
            gpuparticles->particle_angle[poffset] += vc3_cumath::Pi;
            sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
            gpuparticles->particle_pos[poffset] += (-1.00) * gpuparticles->particle_dpos[poffset];
            gpuparticles->particle_dpos[poffset].x = gpuparticles->particle_dpos[poffset].y = 0.00;
            gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
            gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
            /*! Be warned about possible errorous consequent boundary hits !*/
        }
        if (__gpusetup_variables.timedResetTimerType == 0)
        {
            gpuparticles->particle_NRT[poffset] += __gpusetup_variables.timedResetMeanTime;
            /*! Beware of double addition: at boundary hit and here !*/
        }
        else if (__gpusetup_variables.timedResetTimerType == 1)
        {
            flt2 z3 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
            gpuparticles->particle_NRT[poffset] = gpuvariables->currentTime - log(z3) * __gpusetup_variables.timedResetMeanTime;
        }
    }

    // $ homeCheck()
    // =============
    /* Left empty until further times when anything else works and scales good */

    // $ relaxBeta()
    // =============
    // 1. Normal relaxation to zero
    //_deltaBeta *= _deltaBetaDecayRate;
    gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset] +
        (gpuparticles->particle_beta[poffset] - gpuparticles->particle_beta0[poffset]) * __gpuprecomputes.betaDecayRate;

    // 2. Reset to zero after boundary hit
    if (gpuparticles->particle_flag_boundaryHit[poffset])
    {
        if (__gpusetup_variables.boundaryResetType == 0)
        { // 0 - position to the center + random direction
            gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset];
        }
        else if (__gpusetup_variables.boundaryResetType == 1)
        { // 1 - direction to the center
            gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset] * __gpusetup_variables.betaBoundaryMult;
        }
        else if (__gpusetup_variables.boundaryResetType == 2)
        { // 2 - reverse direction + one timestep backward
            gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset] * __gpusetup_variables.betaBoundaryMult;
        }
    }

    // 3. Reset to zero after target hit
    if (gpuparticles->particle_flag_targetHit[poffset] >= 0 && !gpuparticles->particle_flag_boundaryHit[poffset])
    {
        if (gputargets->target_resetType[gpuparticles->particle_flag_targetHit[poffset]] == 0)
        { // 0 - position to the center + random direction
            gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset];
            //printf("relax beta: step=%lld, time=%.15le, pID=%d, beta=%.8le\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset, gpuparticles->particle_beta[poffset]);
        }
        else if (gputargets->target_resetType[gpuparticles->particle_flag_targetHit[poffset]] == 1)
        { // 1 - direction to the center
            gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset] * 
                gputargets->target_betaMult[gpuparticles->particle_flag_targetHit[poffset]];
            //printf("relax beta: step=%lld, time=%.15le, pID=%d, beta=%.8le\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset, gpuparticles->particle_beta[poffset]);
        }
        else if (gputargets->target_resetType[gpuparticles->particle_flag_targetHit[poffset]] == 2)
        { // 2 - reverse direction + revert position one timestep backward
            gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset] *
                gputargets->target_betaMult[gpuparticles->particle_flag_targetHit[poffset]];
            //printf("relax beta: step=%lld, time=%.15le, pID=%d, beta=%.8le\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset, gpuparticles->particle_beta[poffset]);
        }
    }

    // 4. Reset to zero after stochastic reset
    if (gpuparticles->particle_flag_boundaryHit[poffset])
    {
        gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset];
        //printf("relax beta BH: step=%lld, time=%.15le, pID=%d, beta=%.8le\n", gpuvariables->currentStep, gpuvariables->currentTime, poffset, gpuparticles->particle_beta[poffset]);
    }

    /*/ 5. Reset to zero after home hit
    if (_homeHit)
    {
        _deltaBeta = 0.00;
    }*/
}


#endif // VC3_PHYS_KSNP_KERNELS_MOVEPARTICLES
