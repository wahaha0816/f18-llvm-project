! This test checks lowering of OpenMP Master Directive to FIR Dialect.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program main

!$OMP PARALLEL
print*, "Parallel region"
!FIRDialect: omp.parallel {
!FIRDialect: fir.call @_FortranAioBeginExternalListOutput
!FIRDialect: fir.call @_FortranAioOutputAscii
!FIRDialect: fir.call @_FortranAioEndIoStatement

!$OMP MASTER
!FIRDialect: omp.master {
!FIRDialect: fir.call @_FortranAioBeginExternalListOutput
!FIRDialect: fir.call @_FortranAioOutputAscii
!FIRDialect: fir.call @_FortranAioEndIoStatement
!FIRDialect:    omp.terminator
!FIRDialect: }
print*, "Master region"
!$OMP END MASTER

!FIRDialect: omp.terminator
!FIRDialect:  }

!$OMP END PARALLEL
end
