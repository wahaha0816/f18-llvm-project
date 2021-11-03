! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPimplied_iters_allocatable(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>,
! CHECK-SAME %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine implied_iters_allocatable(thing, a1)
  ! No dependence between lhs and rhs.
  ! Lhs may need to be reallocated to conform.
  real :: a1(:)
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(:)
  integer :: i
  
  forall (i=5:13)
  ! commenting out this test for the moment
  !  thing(i)%arr = a1
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine implied_iters_allocatable

! CHECK-LABEL: func @_QPconflicting_allocatable(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFconflicting_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, %[[VAL_1:.*]]: !fir.ref<i32>, %[[VAL_2:.*]]: !fir.ref<i32>) {
subroutine conflicting_allocatable(thing, lo, hi)
  ! Introduce a crossing dependence to incite a (deep) copy.
  integer :: lo,hi
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(:)
  integer :: i
  
  forall (i = lo:hi)
  ! commenting out this test for the moment
  !  thing(i)%arr = thing(hi-i)%arr
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine conflicting_allocatable

! CHECK-LABEL: func @_QPforall_pointer_assign(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>,
! CHECK-SAME: %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>>> {fir.target},
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<i32>, %[[VAL_3:.*]]: !fir.ref<i32>) {
subroutine forall_pointer_assign(ap, at, ii, ij)
  ! Set pointer members in an array of derived type to targets.
  ! No conflicts (multiple-assignment being forbidden, of course).
  type t
     real, pointer :: ptr(:)
  end type t
  type u
     real :: targ(20)
  end type u
  type(t) :: ap(:)
  type(u), target :: at(:)
  integer :: ii, ij

  forall (i = ii:ij:8)
  ! commenting out this test for the moment
  !   ap(i)%ptr => at(i-4)%targ
  end forall
  ! CHECK: return
  ! CHECK: }  
end subroutine forall_pointer_assign

! CHECK-LABEL: func @_QPslice_with_explicit_iters() {
subroutine slice_with_explicit_iters
  ! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10xi32> {bindc_name = "a", uniq_name = "_QFslice_with_explicit_itersEa"}
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 5 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.array<10x10xi32>
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
  ! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
  ! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : i64
  ! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
  ! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
  ! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_21]] : index
  ! CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_25:.*]] = arith.divsi %[[VAL_24]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_25]], %[[VAL_20]] : index
  ! CHECK:           %[[VAL_27:.*]] = select %[[VAL_26]], %[[VAL_25]], %[[VAL_20]] : index
  ! CHECK:           %[[VAL_28:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_29:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_30:.*]] = arith.subi %[[VAL_27]], %[[VAL_28]] : index
  ! CHECK:           %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_13]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_35:.*]] = arith.constant 0 : i32
  ! CHECK:             %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_34]] : i32
  ! CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
  ! CHECK:             %[[VAL_39:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
  ! CHECK:             %[[VAL_41:.*]] = arith.muli %[[VAL_32]], %[[VAL_40]] : index
  ! CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_38]], %[[VAL_41]] : index
  ! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
  ! CHECK:             %[[VAL_46:.*]] = fir.array_update %[[VAL_33]], %[[VAL_36]], %[[VAL_42]], %[[VAL_45]] {Fortran.offsets} : (!fir.array<10x10xi32>, i32, index, index) -> !fir.array<10x10xi32>
  ! CHECK:             fir.result %[[VAL_46]] : !fir.array<10x10xi32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_47:.*]] : !fir.array<10x10xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_48:.*]] to %[[VAL_3]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10xi32>>

  integer :: a(10,10)
  forall (i=1:5)
     a(1:i, i) = -i
  end forall
  ! CHECK:         return
  ! CHECK:       }
end subroutine slice_with_explicit_iters

! CHECK-LABEL: func @_QPembox_argument_with_slice(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<1xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<2x2xi32>>) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (!fir.array<1xi32>) {
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_4]] : index
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
! CHECK:           %[[VAL_24:.*]] = fir.undefined index
! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_26:.*]] = fir.slice %[[VAL_18]], %[[VAL_20]], %[[VAL_21]], %[[VAL_23]], %[[VAL_24]], %[[VAL_24]] : (index, index, i64, i64, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_27:.*]] = fir.embox %[[VAL_1]](%[[VAL_25]]) {{\[}}%[[VAL_26]]] : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[VAL_28:.*]] = fir.call @_QPe(%[[VAL_27]]) : (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_17]] : i32
! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
! CHECK:           %[[VAL_33:.*]] = fir.array_update %[[VAL_15]], %[[VAL_29]], %[[VAL_32]] {Fortran.offsets} : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
! CHECK:           fir.result %[[VAL_33]] : !fir.array<1xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_34:.*]] to %[[VAL_0]] : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
subroutine embox_argument_with_slice(a,b)
  interface
     pure integer function e(a)
       integer, intent(in) :: a(:)
     end function e
  end interface
  integer a(1), b(2,2)

  forall (i=1:1)
     a(i) = e(b(:,i)) + 1
  end forall
! CHECK:         return
! CHECK:       }
end subroutine embox_argument_with_slice
