      subroutine GasChemistry(t_in, t_out) !(0.0, dt_sec)
      use module_data_mosaic_main
      use module_data_mosaic_gas
      USE mcm_Initialize
      use mcm_Integrator
      USE mcm_Model
      use mcm_Global
      
      implicit none 
     
      real(r8) :: t_in, t_out
      real(r8) :: WaterVapor
      REAL(kind=dp) :: T, DVAL(NSPEC)
      REAL(kind=dp) :: RSTATE(20)
      INTEGER :: i, flag_t 

      STEPMIN = 0.0d0
      STEPMAX = 0.0d0
      
      DO i=1,NVAR
         RTOL(i) = 1.0d-4
          ATOL(i) = 1.0d-3
      END DO

      CALL Initialize() 
       TSTART = t_in  
       TEND = trun_ss 
       DT = dt_sec 
       TEMP = te
       
       IF (flag_t /= 1) THEN 
          T = TSTART
       END IF
       flag_t = 1

            TIME = T 

              CALL Update_RCONST()

              CALL INTEGRATE( TIN = T, TOUT = T+DT, RSTATUS_U = RSTATE, &
              ICNTRL_U = (/ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 /) )
              T = RSTATE(1) !tout = tin + dt   

             !CALL SaveData() 
              !call map_mcm_to_mosaic(size(mcm_species), size(mosaic_species))

      end subroutine GasChemistry
