MODULE mcm_Model

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!  Completely defines the model mcm
!    by using all the associated modules
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  USE mcm_Precision
  USE mcm_Parameters
  USE mcm_Global
  USE mcm_Function
  USE mcm_Integrator
  USE mcm_Rates
  USE mcm_Jacobian
 ! USE mcm_Hessian
  USE mcm_Stoichiom
  USE mcm_LinearAlgebra
  USE mcm_Monitor
  USE mcm_Util

END MODULE mcm_Model

