//===-- VectorSubscripts.h -- vector subscripts tools -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines a compiler internal representation for lowered designators
///  containing vector subscripts. This representation allows working on such
///  designators in custom ways while ensuring the designator subscripts are
///  only evaluated once. It is mainly intended for cases that do not fit in
///  the array expression lowering framework like input IO in presence of
///  vector subscripts.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_VECTORSUBSCRIPTS_H
#define FORTRAN_LOWER_VECTORSUBSCRIPTS_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Common/indirection.h"

namespace fir {
class FirOpBuilder;
}

namespace Fortran {

namespace evaluate {
template <typename>
class Expr;
template <typename>
class Designator;
struct SomeType;
} // namespace evaluate

namespace lower {

class AbstractConverter;
class StatementContext;
class LoweredExpr;
class ExplicitIterSpace;

/// MaterializedExpr is a lowered representation primarily meant for to represent lowered Designator<T>. It can also be used to as a wrapper over any fir::ExtendedValue to work with it like a designator.
/// The fundamental aspect is that the values of a MaterializedExpr already exist somewhere in memory or in SSA registers, and
/// that the remaining lowering task is to address them.
///
///
/// A designator `x%a(i,j)%b(1:foo():1, vector, k)%c%d(m)%e1
/// Is lowered into:
///   - an ExtendedValue for ranked base (x%a(i,j)%b)
///   - mlir:Values and ExtendedValues for the triplet, vector subscript and
///     scalar subscripts of the ranked array reference (1:foo():1, vector, k)
///   - a list of fir.field_index and scalar integers mlir::Value for the
///   component
///     path at the right of the ranked array ref (%c%d(m)%e).
///
/// This representation allows later creating loops over the designator elements
/// and fir.array_coor to get the element addresses without re-evaluating any
/// sub-expressions.
class MaterializedExpr {
public:

  struct LoweredVectorSubscript {
    Fortran::lower::LoweredExpr& getVector();
    const Fortran::lower::LoweredExpr& getVector() const;
    /// Copy assignments needed to allow lambda capture.
    LoweredVectorSubscript(const LoweredVectorSubscript&);
    LoweredVectorSubscript(LoweredVectorSubscript&&);
    LoweredVectorSubscript &operator=(const LoweredVectorSubscript &);
    LoweredVectorSubscript &operator=(LoweredVectorSubscript &&);
    LoweredVectorSubscript(Fortran::lower::LoweredExpr&& vector, mlir::Value size);
    ~LoweredVectorSubscript();
    // Lowered vector expression 
    Fortran::common::Indirection<Fortran::lower::LoweredExpr, /*copy-able*/true> vector;
    // Vector size, guaranteed to be of indexType.
    mlir::Value size;
  };

  struct LoweredTriplet {
    // Triplets value, guaranteed to be of indexType.
    mlir::Value lb;
    mlir::Value ub;
    mlir::Value stride;
  };

  using LoweredSubscript =
      std::variant<mlir::Value, LoweredTriplet, LoweredVectorSubscript>;
  using MaybeSubstring = llvm::SmallVector<mlir::Value, 2>;

  explicit MaterializedExpr(const fir::ExtendedValue& exv) : loweredBase{exv}, elementType{fir::getElementType(exv)} {}
  explicit MaterializedExpr(fir::ExtendedValue&& exv) : loweredBase{std::move(exv)}, elementType{fir::getElementType(exv)} {}

  MaterializedExpr(
      fir::ExtendedValue &&loweredBase,
      llvm::SmallVector<LoweredSubscript, 4> &&loweredSubscripts,
      llvm::SmallVector<mlir::Value> &&componentPath,
      MaybeSubstring substringBounds, mlir::Type elementType)
      : loweredBase{std::move(loweredBase)}, loweredSubscripts{std::move(
                                                 loweredSubscripts)},
        componentPath{std::move(componentPath)},
        substringBounds{substringBounds}, elementType{elementType} {};

  MaterializedExpr(
      fir::ArrayLoadOp load, llvm::SmallVector<mlir::Value> &&preRankedPath,
      llvm::SmallVector<LoweredSubscript, 4> &&loweredSubscripts,
      llvm::SmallVector<mlir::Value> &&componentPath,
      MaybeSubstring substringBounds, mlir::Type elementType)
      : loweredBase{std::in_place_type<fir::ArrayLoadOp>, load}, preRankedPath{std::move(preRankedPath)}, loweredSubscripts{std::move(
                                                 loweredSubscripts)},
        componentPath{std::move(componentPath)},
        substringBounds{substringBounds}, elementType{elementType}, readyForAddressing{true} {};

  /// Return the type of the elements of the array section.
  mlir::Type getElementType() const { return elementType; }

  /// Return the fir.array type if this is an array or
  /// the element type for scalars.
  mlir::Type getBaseType() const;

  /// Get extents, empty for scalars.
  llvm::SmallVector<mlir::Value> getExtents(fir::FirOpBuilder& builder, mlir::Location loc) const;

  /// Get lower bounds, empty if scalar or if all-ones.
  llvm::SmallVector<mlir::Value> getLBounds(fir::FirOpBuilder& builder, mlir::Location loc) const;

  /// Get type parameters if this is a character designator or a derived type with length parameters. Return empty vector otherwise.
  llvm::SmallVector<mlir::Value> getTypeParams(fir::FirOpBuilder &builder, mlir::Location loc) const;

  /// Returns an fir::ExtendedValue representing the variable without making a temp.
  /// Cannot be called for variable with vector subscripts.
  /// Will generate a fir.embox or fir.rebox for ArraySection.
  fir::ExtendedValue getAsExtendedValue(fir::FirOpBuilder& builder, mlir::Location loc) const;


  bool hasVectorSubscripts() const;

  bool isArray() const;

  bool baseIsArrayLoad() const {
    return std::holds_alternative<fir::ArrayLoadOp>(loweredBase);
  }

  void prepareForAddressing(fir::FirOpBuilder& builder, mlir::Location loc, bool loadArrays);
  /// Get variable element given implied shape indices. If the variable
  /// is a scalar, the indices are ignored, and the same value is always
  /// returned.
  fir::ExtendedValue getElementAt(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::ValueRange indices) const;

  mlir::Value genArrayUpdateAt(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::Value newElementValue,
                                  mlir::ValueRange indices, mlir::Value previousArrayValue) const;

  mlir::Value genArrayFetchAt(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::ValueRange indices) const;

  /// Create sliceOp for the designator.
  mlir::Value createSlice(fir::FirOpBuilder &builder, mlir::Location loc) const;

  /// Create shapeOp for the designator.
  mlir::Value createShape(fir::FirOpBuilder &builder, mlir::Location loc) const;

  fir::ExtendedValue asBox(fir::FirOpBuilder& builder, mlir::Location loc) const;

  /// Get the base fir::ExtendedValue if this is not an ArrayLoad.
  const fir::ExtendedValue* getBaseIfExtendedValue() const {
    return std::get_if<fir::ExtendedValue>(&loweredBase);
  }


private:

  /// Is this simply an ExtendedValue wrapped as a MaterializedExpr ?
  bool isExtendedValue() const {
    return !baseIsArrayLoad() && preRankedPath.empty() && loweredSubscripts.empty() && componentPath.empty() && substringBounds.empty();
  }

  fir::ExtendedValue genArrayCoor(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::ValueRange indices) const;
  /// Generate the path to be used in fir.array_fetch and fir.array_update, and indicate if it is using non zero based offsets.
  std::pair<llvm::SmallVector<mlir::Value>, bool> genPath(fir::FirOpBuilder& builder, mlir::Location loc, mlir::ValueRange indices) const;

  /// Lowered base of the ranked array ref.
  std::variant<fir::ExtendedValue, fir::ArrayLoadOp> loweredBase;

  /// Scalar subscripts and components at the left of the ranked
  /// array ref (only intended to cover forall raising).
  llvm::SmallVector<mlir::Value> preRankedPath;

  /// Subscripts values of the rank arrayRef part.
  llvm::SmallVector<LoweredSubscript, 4> loweredSubscripts;
  /// Scalar subscripts and components at the right of the ranked
  /// array ref part.
  llvm::SmallVector<mlir::Value, 4> componentPath;
  /// List of substring bounds if this is a substring (only the lower bound if
  /// the upper is implicit).
  MaybeSubstring substringBounds;
  /// Type of the elements described by this array section.
  mlir::Type elementType;

  mlir::Value shape;
  mlir::Value slice;
  // shape/slice created if needed, allocatable/pointer unwrapped.
  bool readyForAddressing = false;
};

/// Lower a Designator<T> to a MaterializedExpr.
template<typename T>
struct DesignatorBuilder {
  static MaterializedExpr gen(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Designator<T> &expr, ExplicitIterSpace* explicitIterSpace, bool loadArrays);
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_VECTORSUBSCRIPTS_H
