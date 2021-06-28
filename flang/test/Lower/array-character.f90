! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPissue
! CHECK-SAME: %[[C1:.*]]: !fir.boxchar<1>, %[[C2:.*]]: !fir.boxchar<1>
subroutine issue(c1, c2)
  ! CHECK-DAG:  %[[VAL_0:.*]] = constant 3 : index
  ! CHECK-DAG:  %[[VAL_1:.*]] = constant 4 : index
  ! CHECK-DAG:  %[[VAL_2:.*]] = constant 0 : index
  ! CHECK-DAG:  %[[VAL_3:.*]] = constant 1 : index
  ! CHECK-DAG:  %[[VAL_4:.*]]:2 = fir.unboxchar %[[C1:.*]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG:  %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK-DAG:  %[[VAL_7:.*]]:2 = fir.unboxchar %[[C2:.*]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG:  %[[VAL_9:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
  ! CHECK:  %[[VAL_10:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
  ! CHECK:  br ^bb1(%[[VAL_2]], %[[VAL_0]] : index, index)
  ! CHECK:^bb1(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
  ! CHECK:  %[[VAL_13:.*]] = cmpi sgt, %[[VAL_12]], %[[VAL_2]] : index
  ! CHECK:  cond_br %[[VAL_13]], ^bb2, ^bb9
  ! CHECK:^bb2:
  ! CHECK:  %[[VAL_14:.*]] = addi %[[VAL_11]], %[[VAL_3]] : index
  ! CHECK:  %[[VAL_15:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_10]]) %[[VAL_14]] typeparams %[[VAL_7]]#1 : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:  %[[VAL_16:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_10]]) %[[VAL_14]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,4>>
  ! CHECK:  br ^bb3(%[[VAL_2]], %[[VAL_1]] : index, index)
  ! CHECK:^bb3(%[[VAL_17:.*]]: index, %[[VAL_18:.*]]: index):
  ! CHECK:  %[[VAL_19:.*]] = cmpi sgt, %[[VAL_18]], %[[VAL_2]] : index
  ! CHECK:  cond_br %[[VAL_19]], ^bb4, ^bb8
  ! CHECK:^bb4:
  ! CHECK:  %[[VAL_20:.*]] = cmpi slt, %[[VAL_17]], %[[VAL_7]]#1 : index
  ! CHECK:  cond_br %[[VAL_20]], ^bb5, ^bb6
  ! CHECK:^bb5:
  ! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_21]], %[[VAL_17]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.char<1>>
  ! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_17]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  fir.store %[[VAL_23]] to %[[VAL_25]] : !fir.ref<!fir.char<1>>
  ! CHECK:  br ^bb7
  ! CHECK:^bb6:
  ! CHECK:  %[[VAL_26:.*]] = fir.string_lit [32 : i8](1) : !fir.char<1>
  ! CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_28:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_17]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  fir.store %[[VAL_26]] to %[[VAL_28]] : !fir.ref<!fir.char<1>>
  ! CHECK:  br ^bb7
  ! CHECK:^bb7:
  ! CHECK:  %[[VAL_29:.*]] = addi %[[VAL_17]], %[[VAL_3]] : index
  ! CHECK:  %[[VAL_30:.*]] = subi %[[VAL_18]], %[[VAL_3]] : index
  ! CHECK:  br ^bb3(%[[VAL_29]], %[[VAL_30]] : index, index)
  ! CHECK:^bb8:
  ! CHECK:  %[[VAL_31:.*]] = subi %[[VAL_12]], %[[VAL_3]] : index
  ! CHECK:  br ^bb1(%[[VAL_14]], %[[VAL_31]] : index, index)
  ! CHECK:^bb9:
  ! CHECK:  return
  character(4) :: c1(3)
  character(*) :: c2(3)
  c1 = c2
end subroutine

! CHECK-LABEL: func @_QPissu857_a(
! CHECK-SAME: %[[X:.*]]: !fir.boxchar<1>, %[[Y:.*]]: !fir.boxchar<1>) {
subroutine issu857_a(x, y)
  character(*) :: x(4), y
  ! CHECK-DAG:  %[[VAL_0:.*]] = constant 4 : index
  ! CHECK-DAG:  %[[VAL_1:.*]] = constant 0 : index
  ! CHECK-DAG:  %[[VAL_2:.*]] = constant 1 : index
  ! CHECK-DAG:  %[[VAL_3:.*]]:2 = fir.unboxchar %[[X:.*]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG:  %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<4x!fir.char<1,?>>>
  ! CHECK-DAG:  %[[VAL_6:.*]]:2 = fir.unboxchar %[[Y:.*]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
  ! CHECK:  br ^bb1(%[[VAL_1]], %[[VAL_0]] : index, index)
  ! CHECK:^bb1(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
  ! CHECK:  %[[VAL_11:.*]] = cmpi sgt, %[[VAL_10]], %[[VAL_1]] : index
  ! CHECK:  cond_br %[[VAL_11]], ^bb2, ^bb9
  ! CHECK:^bb2:
  ! CHECK:  %[[VAL_12:.*]] = addi %[[VAL_9]], %[[VAL_2]] : index
  ! CHECK:  %[[VAL_13:.*]] = fir.array_coor %[[VAL_5]](%[[VAL_8]]) %[[VAL_12]] typeparams %[[VAL_3]]#1 : (!fir.ref<!fir.array<4x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:  br ^bb3(%[[VAL_1]], %[[VAL_3]]#1 : index, index)
  ! CHECK:^bb3(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
  ! CHECK:  %[[VAL_16:.*]] = cmpi sgt, %[[VAL_15]], %[[VAL_1]] : index
  ! CHECK:  cond_br %[[VAL_16]], ^bb4, ^bb8
  ! CHECK:^bb4:
  ! CHECK:  %[[VAL_17:.*]] = cmpi slt, %[[VAL_14]], %[[VAL_6]]#1 : index
  ! CHECK:  cond_br %[[VAL_17]], ^bb5, ^bb6
  ! CHECK:^bb5:
  ! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_18]], %[[VAL_14]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.char<1>>
  ! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_21]], %[[VAL_14]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  fir.store %[[VAL_20]] to %[[VAL_22]] : !fir.ref<!fir.char<1>>
  ! CHECK:  br ^bb7
  ! CHECK:^bb6:
  ! CHECK:  %[[VAL_23:.*]] = fir.string_lit [32 : i8](1) : !fir.char<1>
  ! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_14]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  fir.store %[[VAL_23]] to %[[VAL_25]] : !fir.ref<!fir.char<1>>
  ! CHECK:  br ^bb7
  ! CHECK:^bb7:
  ! CHECK:  %[[VAL_26:.*]] = addi %[[VAL_14]], %[[VAL_2]] : index
  ! CHECK:  %[[VAL_27:.*]] = subi %[[VAL_15]], %[[VAL_2]] : index
  ! CHECK:  br ^bb3(%[[VAL_26]], %[[VAL_27]] : index, index)
  ! CHECK:^bb8:
  ! CHECK:  %[[VAL_28:.*]] = subi %[[VAL_10]], %[[VAL_2]] : index
  ! CHECK:  br ^bb1(%[[VAL_12]], %[[VAL_28]] : index, index)
  ! CHECK:^bb9:
  ! CHECK:  return
  x = y
end subroutine

! CHECK-LABEL: func @_QPissue857_b(
! CHECK-SAME: %[[X:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
! CHECK-SAME: %[[Y:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>)
subroutine issue857_b(x, y)
  character(*) :: x(:)
  character(*) :: y(:)
  ! CHECK-DAG:  %[[VAL_0:.*]] = constant 0 : index
  ! CHECK-DAG:  %[[VAL_1:.*]] = constant 1 : index
  ! CHECK:  %[[VAL_2:.*]]:3 = fir.box_dims %[[X]], %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
  ! CHECK:  br ^bb1(%[[VAL_0]], %[[VAL_2]]#1 : index, index)
  ! CHECK:^bb1(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
  ! CHECK:  %[[VAL_6:.*]] = cmpi sgt, %[[VAL_5]], %[[VAL_0]] : index
  ! CHECK:  cond_br %[[VAL_6]], ^bb2, ^bb9
  ! CHECK:^bb2:
  ! CHECK:  %[[VAL_7:.*]] = addi %[[VAL_4]], %[[VAL_1]] : index
  ! CHECK:  %[[VAL_8:.*]] = fir.array_coor %[[Y]] %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:  %[[VAL_10:.*]] = fir.box_elesize %[[Y]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  ! CHECK:  %[[VAL_11:.*]] = fir.box_elesize %[[X]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  ! CHECK:  %[[VAL_12:.*]] = fir.array_coor %[[X]] %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:  br ^bb3(%[[VAL_0]], %[[VAL_11]] : index, index)
  ! CHECK:^bb3(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
  ! CHECK:  %[[VAL_15:.*]] = cmpi sgt, %[[VAL_14]], %[[VAL_0]] : index
  ! CHECK:  cond_br %[[VAL_15]], ^bb4, ^bb8
  ! CHECK:^bb4:
  ! CHECK:  %[[VAL_16:.*]] = cmpi slt, %[[VAL_13]], %[[VAL_10]] : index
  ! CHECK:  cond_br %[[VAL_16]], ^bb5, ^bb6
  ! CHECK:^bb5:
  ! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_13]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<!fir.char<1>>
  ! CHECK:  %[[VAL_20:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_20]], %[[VAL_13]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  fir.store %[[VAL_19]] to %[[VAL_21]] : !fir.ref<!fir.char<1>>
  ! CHECK:  br ^bb7
  ! CHECK:^bb6:
  ! CHECK:  %[[VAL_22:.*]] = fir.string_lit [32 : i8](1) : !fir.char<1>
  ! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:  %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_23]], %[[VAL_13]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:  fir.store %[[VAL_22]] to %[[VAL_24]] : !fir.ref<!fir.char<1>>
  ! CHECK:  br ^bb7
  ! CHECK:^bb7:
  ! CHECK:  %[[VAL_25:.*]] = addi %[[VAL_13]], %[[VAL_1]] : index
  ! CHECK:  %[[VAL_26:.*]] = subi %[[VAL_14]], %[[VAL_1]] : index
  ! CHECK:  br ^bb3(%[[VAL_25]], %[[VAL_26]] : index, index)
  ! CHECK:^bb8:
  ! CHECK:  %[[VAL_27:.*]] = subi %[[VAL_5]], %[[VAL_1]] : index
  ! CHECK:  br ^bb1(%[[VAL_7]], %[[VAL_27]] : index, index)
  ! CHECK:^bb9:
  ! CHECK:  return

  x = y
end subroutine

program p
  character(4) :: c1(3)
  character(4) :: c2(3) = ["abcd", "    ", "    "]
  print *, c2
  call issue(c1, c2)
  print *, c1
end program p

