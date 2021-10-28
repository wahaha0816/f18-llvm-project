! Test array-value-copy
  
! RUN: bbc %s -o - | FileCheck %s

! Copy not needed
! CHECK-LABEL: func @_QPtest1(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:   ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:     fir.freemem %
! CHECK:         return
! CHECK:       }
subroutine test1(a)
  integer :: a(3)

  a = a + 1
end subroutine test1

! Copy not needed
! CHECK-LABEL: func @_QPtest2(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:   ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:     fir.freemem %
! CHECK:         return
! CHECK:       }
subroutine test2(a, b)
  integer :: a(3), b(3)

  a = b + 1
end subroutine test2

! Copy not needed
! CHECK-LABEL: func @_QPtest3(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:   ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:     fir.freemem %
! CHECK:         return
! CHECK:       }
subroutine test3(a)
  integer :: a(3)

  forall (i=1:3)
     a(i) = a(i) + 1
  end forall
end subroutine test3

! Make a copy. (Crossing dependence)
! CHECK-LABEL: func @_QPtest4(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:         fir.freemem %{{.*}} : !fir.heap<!fir.array<3xi32>>
! CHECK:         return
! CHECK:       }
subroutine test4(a)
  integer :: a(3)

  forall (i=1:3)
     a(i) = a(4-i) + 1
  end forall
end subroutine test4

! Make a copy. (Carried dependence)
! CHECK-LABEL: func @_QPtest5(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:         fir.freemem %{{.*}} : !fir.heap<!fir.array<3xi32>>
! CHECK:         return
! CHECK:       }
subroutine test5(a)
  integer :: a(3)

  forall (i=2:3)
     a(i) = a(i-1) + 14
  end forall
end subroutine test5

! Make a copy. (Carried dependence)
! CHECK-LABEL: func @_QPtest6(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:         fir.freemem %{{.*}} : !fir.heap<!fir.array<3x!fir.type<_QFtest6Tt{m:!fir.array<3xi32>}>>>
! CHECK:         return
! CHECK:       }
subroutine test6(a)
  type t
     integer :: m(3)
  end type t
  type(t) :: a(3)

  forall (i=2:3)
     a(i)%m = a(i-1)%m + 14
  end forall
end subroutine test6

! Make a copy. (Overlapping partial CHARACTER update.)
! CHECK-LABEL: func @_QPtest7(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:         fir.freemem %{{.*}} : !fir.heap<!fir.array<3x!fir.char<1,8>>>
! CHECK:         return
! CHECK:       }
subroutine test7(a)
  character(8) :: a(3)

  a(:)(2:5) = a(:)(3:6)
end subroutine test7

! Do not make a copy.
! CHECK-LABEL: func @_QPtest8(
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK:       ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:   ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! CHECK-NOT:     fir.freemem %
! CHECK:         return
! CHECK:       }
subroutine test8(a,b)
  character(8) :: a(3), b(3)

  a(:)(2:5) = b(:)(3:6)
end subroutine test8
