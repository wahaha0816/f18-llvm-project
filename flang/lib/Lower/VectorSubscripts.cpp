//===-- VectorSubscripts.cpp -- Vector subscripts tools -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "IterationSpace.h"
#include "flang/Lower/VectorSubscripts.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Semantics/expression.h"

namespace {

//===--------------------------------------------------------------------===//
// MaterializedExpr generation from Designator<T>
//===--------------------------------------------------------------------===//

static fir::ExtendedValue addressComponents(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& exv, llvm::ArrayRef<fir::FieldIndexOp> fields) {
  TODO(loc, "address components");
}

static constexpr bool mustGenArrayCoor = false;


static fir::ExtendedValue genArrayCoor(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  auto addr = fir::getBase(array);
  auto eleTy = fir::unwrapSequenceType(fir::unwrapPassByRefType(addr.getType()));
  auto eleRefTy = builder.getRefType(eleTy);
  auto shape = builder.createShape(loc, array);
  auto elementAddr = builder.create<fir::ArrayCoorOp>(
      loc, eleRefTy, addr, shape, /*slice=*/mlir::Value{}, coordinates,
      fir::getTypeParams(array));
  return fir::factory::arrayElementToExtendedValue(builder, loc, array,
                                                   elementAddr);
}

static llvm::SmallVector<mlir::Value> toZeroBasedIndices(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  llvm::SmallVector<mlir::Value> zeroBased;
  auto one = builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  for (auto coor : llvm::enumerate(coordinates)) {
    auto lb = fir::factory::readLowerBound(builder, loc, array, coor.index(), one);
    auto ty = coor.value().getType();
    lb = builder.createConvert(loc, ty, lb);
    zeroBased.push_back(builder.create<mlir::arith::SubIOp>(loc, ty, coor.value(), lb));
  }
  return zeroBased;
}

/// Compute the shape of a slice (TODO share in fir::factory)
static llvm::SmallVector<mlir::Value> computeSliceShape(fir::FirOpBuilder& builder, mlir::Location loc, mlir::Value slice) {
  llvm::SmallVector<mlir::Value> slicedShape;
  auto slOp = mlir::cast<fir::SliceOp>(slice.getDefiningOp());
  auto triples = slOp.triples();
  auto idxTy = builder.getIndexType();
  for (unsigned i = 0, end = triples.size(); i < end; i += 3) {
    if (!mlir::isa_and_nonnull<fir::UndefOp>(
            triples[i + 1].getDefiningOp())) {
      // (..., lb:ub:step, ...) case:  extent = max((ub-lb+step)/step, 0)
      // See Fortran 2018 9.5.3.3.2 section for more details.
      auto res = builder.genExtentFromTriplet(loc, triples[i], triples[i + 1],
                                              triples[i + 2], idxTy);
      slicedShape.emplace_back(res);
    } else {
      // do nothing. `..., i, ...` case, so dimension is dropped.
    }
  }
  return slicedShape;
}

static llvm::SmallVector<mlir::Value> getShape(fir::FirOpBuilder& builder, mlir::Location loc, fir::ArrayLoadOp array) {
  if (array.slice())
    return computeSliceShape(builder, loc, array.slice());
  if (array.memref().getType().isa<fir::BoxType>())
    return fir::factory::readExtents(builder, loc,
                                     fir::BoxValue{array.memref()});
  auto extents = fir::factory::getExtents(array.shape());
  return {extents.begin(), extents.end()};
}


/// Lower an ArrayRef to a fir.coordinate_of using an element offset instead
/// of array indexes.
/// This generates offset computation from the indexes and length parameters,
/// and use the offset to access the element with a fir.coordinate_of. This
/// must only be used if it is not possible to generate a normal
/// fir.coordinate_of using array indexes (i.e. when the shape information is
/// unavailable in the IR).
fir::ExtendedValue genOffsetAndCoordinateOp(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  auto addr = fir::getBase(array);
  auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto seqTy = builder.getRefType(builder.getVarLenSeqTy(eleTy));
  auto refTy = builder.getRefType(eleTy);
  auto base = builder.createConvert(loc, seqTy, addr);
  auto idxTy = builder.getIndexType();
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
    return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
  };
  auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
    mlir::Value total = zero;
    assert(arr.getExtents().size() == coordinates.size());
    delta = builder.createConvert(loc, idxTy, delta);
    unsigned dim = 0;
    for (auto [ext, sub] : llvm::zip(arr.getExtents(), coordinates)) {
      auto val = builder.createConvert(loc, idxTy, sub);
      auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
      auto diff = builder.create<mlir::arith::SubIOp>(loc, val, lb);
      auto prod = builder.create<mlir::arith::MulIOp>(loc, delta, diff);
      total = builder.create<mlir::arith::AddIOp>(loc, prod, total);
      if (ext)
        delta = builder.create<mlir::arith::MulIOp>(loc, delta, ext);
      ++dim;
    }
    auto origRefTy = refTy;
    if (fir::factory::CharacterExprHelper::isCharacterScalar(refTy)) {
      auto chTy = fir::factory::CharacterExprHelper::getCharacterType(refTy);
      if (fir::characterWithDynamicLen(chTy)) {
        auto ctx = builder.getContext();
        auto kind = fir::factory::CharacterExprHelper::getCharacterKind(chTy);
        auto singleTy = fir::CharacterType::getSingleton(ctx, kind);
        refTy = builder.getRefType(singleTy);
        auto seqRefTy = builder.getRefType(builder.getVarLenSeqTy(singleTy));
        base = builder.createConvert(loc, seqRefTy, base);
      }
    }
    auto coor = builder.create<fir::CoordinateOp>(
        loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    // Convert to expected, original type after address arithmetic.
    return builder.createConvert(loc, origRefTy, coor);
  };
  return array.match(
      [&](const fir::ArrayBoxValue &arr) -> fir::ExtendedValue {
        return genFullDim(arr, one);
      },
      [&](const fir::CharArrayBoxValue &arr) -> fir::ExtendedValue {
        auto delta = arr.getLen();
        // If the length is known in the type, fir.coordinate_of will
        // already take the length into account.
        if (fir::factory::CharacterExprHelper::hasConstantLengthInType(arr))
          delta = one;
        return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
      },
      [&](const fir::BoxValue &arr) -> fir::ExtendedValue {
        // CoordinateOp for BoxValue is not generated here. The dimensions
        // must be kept in the fir.coordinate_op so that potential fir.box
        // strides can be applied by codegen.
        fir::emitFatalError(
            loc, "internal: BoxValue in dim-collapsed fir.coordinate_of");
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "internal: array lowering failed");
      });
}

/// Address an array with user coordinates (not zero based).
static fir::ExtendedValue addressArray(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& array, llvm::ArrayRef<mlir::Value> coordinates) {
  if (mustGenArrayCoor)
    return genArrayCoor(builder, loc, array, coordinates);
  auto base = fir::getBase(array);
  auto baseType = fir::unwrapPassByRefType(base.getType());
  if ((array.rank() > 1 && fir::hasDynamicSize(baseType)) ||
      fir::characterWithDynamicLen(fir::unwrapSequenceType(baseType)))
    if (!array.getBoxOf<fir::BoxValue>())
      return genOffsetAndCoordinateOp(builder, loc, array, coordinates);
  auto eleRefTy = builder.getRefType(fir::unwrapSequenceType(baseType));
  // fir::CoordinateOp is zero based.
  auto zeroBasedIndices = toZeroBasedIndices(builder, loc, array, coordinates);
  auto addr = builder.create<fir::CoordinateOp>(loc, eleRefTy, base, zeroBasedIndices);
  return fir::factory::arrayElementToExtendedValue(builder, loc, array, addr);
}

static fir::ExtendedValue getComplexPart(fir::FirOpBuilder& builder, mlir::Location loc, const fir::ExtendedValue& scalarComplex, mlir::Value part) {
  fir::factory::ComplexExprHelper helper(builder, loc);
  auto base = fir::getBase(scalarComplex);
  auto cmplxType = fir::dyn_cast_ptrEleTy(base.getType());
  assert(cmplxType && "complex variable must be in memory");
  auto eleTy = helper.getComplexPartType(cmplxType);
  mlir::Value result = builder.create<fir::CoordinateOp>(
      loc, builder.getRefType(eleTy), base, mlir::ValueRange{part});
  return result;
}

/// Helper class to lower a designator containing vector subscripts into a
/// lowered representation that can be worked with.
class DesignatorBuilderImpl {
public:
  DesignatorBuilderImpl(mlir::Location loc,
                            Fortran::lower::AbstractConverter &converter,
                            Fortran::lower::StatementContext &stmtCtx, Fortran::lower::ExplicitIterSpace* explicitIterSpace, bool loadArrays)
      : converter{converter}, stmtCtx{stmtCtx}, explicitIterSpace{explicitIterSpace}, loc{loc}, loadArrays{loadArrays} {}

  template <typename T>
  Fortran::lower::MaterializedExpr genDesignator(const Fortran::evaluate::Designator<T> &designator) {
    auto elementType = std::visit([&](const auto &x) { return gen(x); }, designator.u);
    applyLeftAddressingPart();
    if (auto baseExv = std::get_if<fir::ExtendedValue>(&loweredBase)) {
      assert(preRankedPath.empty() && "must have been applied");
      return Fortran::lower::MaterializedExpr(
          std::move(*baseExv), std::move(loweredSubscripts),
          std::move(componentPath), substringBounds, elementType);
    }
    auto arrayLoad = std::get<fir::ArrayLoadOp>(loweredBase);
    return Fortran::lower::MaterializedExpr(
        arrayLoad, std::move(preRankedPath), std::move(loweredSubscripts),
        std::move(componentPath), substringBounds, elementType);
  }


private:
  using LoweredVectorSubscript =
      Fortran::lower::MaterializedExpr::LoweredVectorSubscript;
  using LoweredTriplet = Fortran::lower::MaterializedExpr::LoweredTriplet;
  using LoweredSubscript = Fortran::lower::MaterializedExpr::LoweredSubscript;
  using MaybeSubstring = Fortran::lower::MaterializedExpr::MaybeSubstring;

  // The gen(X) methods visit X to lower its base and subscripts and return the
  // type of X elements.

  mlir::Type gen(const Fortran::evaluate::DataRef &dataRef) {
    return std::visit([&](const auto &ref) -> mlir::Type { return gen(ref); },
                      dataRef.u);
  }

  template<typename T>
  llvm::Optional<mlir::Type> alreadyLoadedInExplicitContext(const T& x) {
    if (explicitIterSpace) {
      if (auto load = explicitIterSpace->findBinding(&x)) {
        lastPartWasRanked = true;
        loweredBase.emplace<fir::ArrayLoadOp>(load);
        return fir::unwrapSequenceType(load.getType());
      }
    }
    return llvm::None;
  }

  mlir::Type gen(const Fortran::evaluate::SymbolRef &symRef) {
    if (auto arrayLoadEleTy = alreadyLoadedInExplicitContext(symRef.get()))
      return arrayLoadEleTy.getValue();
    loweredBase = converter.getSymbolExtendedValue(symRef);
    if (symRef->Rank() > 0)
      lastPartWasRanked = true;
    return fir::getElementType(std::get<fir::ExtendedValue>(loweredBase));
  }

  mlir::Type gen(const Fortran::evaluate::Substring &substring) {
    // StaticDataObject::Pointer bases are constants and cannot be
    // subscripted, so the base must be a DataRef here.
    auto baseElementType =
        gen(std::get<Fortran::evaluate::DataRef>(substring.parent()));
    startRightPartIfLastPartWasRanked();
    auto &builder = converter.getFirOpBuilder();
    auto idxTy = builder.getIndexType();
    auto lb = genScalarValue(substring.lower());
    substringBounds.emplace_back(builder.createConvert(loc, idxTy, lb));
    if (const auto &ubExpr = substring.upper()) {
      auto ub = genScalarValue(*ubExpr);
      substringBounds.emplace_back(builder.createConvert(loc, idxTy, ub));
    }
    return baseElementType;
  }

  mlir::Type gen(const Fortran::evaluate::ComplexPart &complexPart) {
    auto complexType = gen(complexPart.complex());
    startRightPartIfLastPartWasRanked();
    auto &builder = converter.getFirOpBuilder();
    auto i32Ty = builder.getI32Type(); // llvm's GEP requires i32
    auto offset = builder.createIntegerConstant(
        loc, i32Ty,
        complexPart.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    if (isAfterRankedPart)
      componentPath.emplace_back(offset);
    else
      preRankedPath.emplace_back(offset);
    return fir::factory::ComplexExprHelper{builder, loc}.getComplexPartType(
        complexType);
  }

  mlir::Type gen(const Fortran::evaluate::Component &component) {
    fir::RecordType recTy;
    if (auto arrayLoadEleTy = alreadyLoadedInExplicitContext(component))
      recTy = arrayLoadEleTy.getValue().cast<fir::RecordType>();
    else
      recTy = gen(component.base()).cast<fir::RecordType>();
    startRightPartIfLastPartWasRanked();
    const auto &componentSymbol = component.GetLastSymbol();
    // Parent components will not be found here, they are not part
    // of the FIR type and cannot be used in the path yet.
    if (componentSymbol.test(Fortran::semantics::Symbol::Flag::ParentComp))
      TODO(loc, "Reference to parent component");
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    auto componentName = toStringRef(componentSymbol.name());
    // Parameters threading in field_index is not yet very clear. We only
    // have the ones of the ranked array ref at hand, but it looks like
    // the fir.field_index expects the one of the direct base.
    if (recTy.getNumLenParams() != 0)
      TODO(loc, "threading length parameters in field index op");
    auto &builder = converter.getFirOpBuilder();
    auto fieldIndex = builder.create<fir::FieldIndexOp>(
        loc, fldTy, componentName, recTy, /*typeParams*/ llvm::None);
    if (isAfterRankedPart)
      componentPath.emplace_back(fieldIndex);
    else
      preRankedPath.emplace_back(fieldIndex);
    if (componentSymbol.Rank() > 0)
      lastPartWasRanked = true;
    return fir::unwrapSequenceType(recTy.getType(componentName));
  }

  mlir::Type gen(const Fortran::evaluate::ArrayRef &arrayRef) {
    mlir::Type elementType;
    if (auto arrayLoadEleTy = alreadyLoadedInExplicitContext(arrayRef))
      elementType = arrayLoadEleTy.getValue();
    else
      elementType = gen(namedEntityToDataRef(arrayRef.base()));
      
    auto isTripletOrVector =
        [](const Fortran::evaluate::Subscript &subscript) -> bool {
      return std::visit(
          Fortran::common::visitors{
              [](const Fortran::evaluate::IndirectSubscriptIntegerExpr &expr) {
                return expr.value().Rank() != 0;
              },
              [&](const Fortran::evaluate::Triplet &) { return true; }},
          subscript.u);
    };
    if (llvm::any_of(arrayRef.subscript(), isTripletOrVector)) {
      genRankedArrayRefSubscripts(arrayRef);
      return elementType;
    }

    // This is a scalar ArrayRef (only scalar indexes), collect the indexes and
    // visit the base that must contain another arrayRef with the vector
    // subscript.
    for (const auto &subscript : arrayRef.subscript()) {
      const auto &expr =
          std::get<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
              subscript.u);
      auto subscriptValue = genScalarValue(expr.value());
      if (isAfterRankedPart)
        componentPath.emplace_back(subscriptValue);
      else
        preRankedPath.emplace_back(subscriptValue);
    }
    // The last part rank was "consumed" by the subscripts.
    lastPartWasRanked = false;
    return elementType;
  }

  /// Lower the subscripts and base of the ArrayRef that is an array (there must
  /// be one since there is a vector subscript, and there can only be one
  /// according to C925).
  void genRankedArrayRefSubscripts(
      const Fortran::evaluate::ArrayRef &arrayRef) {
    applyLeftAddressingPart();
    // Lower and save the subscripts
    auto &builder = converter.getFirOpBuilder();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    for (const auto &subscript : llvm::enumerate(arrayRef.subscript())) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &expr) {
                if (expr.value().Rank() == 0) {
                  // Simple scalar subscript
                  loweredSubscripts.emplace_back(genScalarValue(expr.value()));
                } else {
                  // Vector subscript.
                  auto vector = Fortran::lower::genExpr(converter, loc,
                      expr.value(), stmtCtx, explicitIterSpace, loadArrays);
                  vector.prepareForAddressing(builder, loc, loadArrays);
                  auto extents = vector.getExtents(builder, loc);
                  assert(extents.size() == 1);
                  auto size = builder.createConvert(loc, idxTy, extents[0]);
                  loweredSubscripts.emplace_back(
                      LoweredVectorSubscript{std::move(vector), size});
                }
              },
              [&](const Fortran::evaluate::Triplet &triplet) {
                mlir::Value lb, ub;
                if (const auto &lbExpr = triplet.lower())
                  lb = genScalarValue(*lbExpr);
                else
                  lb = getLeftPartLowerBound(subscript.index(), one);
                lb = builder.createConvert(loc, idxTy, lb);
                if (const auto &ubExpr = triplet.upper()) {
                  ub = genScalarValue(*ubExpr);
                  ub = builder.createConvert(loc, idxTy, ub);
                } else {
                  // ub = lb + extent -1
                  ub = getLeftPartExtent(subscript.index());
                  ub = builder.createConvert(loc, idxTy, ub);
                  ub = builder.create<mlir::arith::SubIOp>(loc, ub, one);
                  ub = builder.create<mlir::arith::AddIOp>(loc, lb, ub);
                }
                auto stride = genScalarValue(triplet.stride());
                stride = builder.createConvert(loc, idxTy, stride);
                loweredSubscripts.emplace_back(LoweredTriplet{lb, ub, stride});
              },
          },
          subscript.value().u);
    }
    isAfterRankedPart = true;
  }

  mlir::Type gen(const Fortran::evaluate::CoarrayRef &) {
    // Is this possible/legal ?
    TODO(loc, "Coarray ref with vector subscript in IO input");
  }

  void applyLeftAddressingPart() {
    auto* baseExv = std::get_if<fir::ExtendedValue>(&loweredBase);
    if (!baseExv)
      return;
    auto &builder = converter.getFirOpBuilder();
    if (preRankedPath.empty() && substringBounds.empty())
      return;
    auto prePath = preRankedPath.begin();
    while (prePath != preRankedPath.end()) {
      if (prePath->getType().isa<fir::FieldType>()) {
        llvm::SmallVector<fir::FieldIndexOp> fields;
        while (prePath != preRankedPath.end()) {
          auto fieldOp = prePath->getDefiningOp<fir::FieldIndexOp>();
          if (!fieldOp)
            break;
          fields.push_back(fieldOp);
          prePath++;
        }
        ++prePath; 
        *baseExv = addressComponents(builder, loc, *baseExv, fields);
      } else {
        auto rank = baseExv->rank();
        if (rank > 0) {
          llvm::SmallVector<mlir::Value> coors;
          while (prePath != preRankedPath.end() && rank > 0) {
            if (prePath->getType().isa<fir::FieldType>())
              break;
            coors.push_back(*prePath);
            ++prePath;
            rank--;
          }
          assert(rank == 0 && "rank mismatch");
          *baseExv = addressArray(builder, loc, *baseExv, coors);
        } else {
          *baseExv = getComplexPart(builder, loc, *baseExv, *prePath);
          ++prePath;
        }
      }
    }
    preRankedPath.clear();

    // Keep substring info if this is a ranked array section.
    if (!loweredSubscripts.empty())
      return;

    if (!substringBounds.empty()) {
      auto charBox = baseExv->getCharBox();
      assert(charBox && "substring must have character base");
      *baseExv = fir::factory::CharacterExprHelper{builder, loc}.createSubstring(
          *charBox, substringBounds);
      substringBounds.clear();
    }
  }

  template <typename A>
  mlir::Value genScalarValue(const A &expr) {
    // TODO replace by a Fortran::lower::ExprLower::gen call.
    return fir::getBase(converter.genExprValue(toEvExpr(expr), stmtCtx));
  }

  void startRightPartIfLastPartWasRanked() {
    if (lastPartWasRanked)
      isAfterRankedPart = true;
  }

  Fortran::evaluate::DataRef
  namedEntityToDataRef(const Fortran::evaluate::NamedEntity &namedEntity) {
    if (namedEntity.IsSymbol())
      return Fortran::evaluate::DataRef{namedEntity.GetFirstSymbol()};
    return Fortran::evaluate::DataRef{namedEntity.GetComponent()};
  }

  Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  namedEntityToExpr(const Fortran::evaluate::NamedEntity &namedEntity) {
    return Fortran::evaluate::AsGenericExpr(namedEntityToDataRef(namedEntity))
        .value();
  }

  mlir::Value getLeftPartLowerBound(unsigned dim, mlir::Value one) {
    auto* baseExv = std::get_if<fir::ExtendedValue>(&loweredBase);
    if (baseExv) {
      auto &builder = converter.getFirOpBuilder();
      return fir::factory::readLowerBound(builder, loc, *baseExv,
                                      dim, one);
    }
    // FIXME: component array lower bounds are not always ones.
    if (!preRankedPath.empty())
      return one;
    auto load = std::get<fir::ArrayLoadOp>(loweredBase);
    if (load.slice())
      return one;
    auto origins = fir::factory::getOrigins(load.shape());
    if (origins.empty())
      return one;
    assert(origins.size() > dim);
    return origins[dim];
  }

  mlir::Value getLeftPartExtent(unsigned dim) {
    auto &builder = converter.getFirOpBuilder();
    auto* baseExv = std::get_if<fir::ExtendedValue>(&loweredBase);
    if (baseExv) {
      return fir::factory::readExtent(builder, loc, *baseExv,
                                                   dim);
    }
    if (!preRankedPath.empty())
      TODO(loc, "getLeftPartExtent arrayLoad with preRankedPath");
    auto shape = getShape(builder, loc, std::get<fir::ArrayLoadOp>(loweredBase));
    assert(shape.size() > dim);
    return shape[dim];
  }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::ExplicitIterSpace* explicitIterSpace;
  mlir::Location loc;
  bool loadArrays;
  /// Elements of the designators being built.
  std::variant<fir::ExtendedValue, fir::ArrayLoadOp> loweredBase;
  llvm::SmallVector<mlir::Value> preRankedPath;
  llvm::SmallVector<LoweredSubscript, 4> loweredSubscripts;
  llvm::SmallVector<mlir::Value> componentPath;
  MaybeSubstring substringBounds;
  bool isAfterRankedPart = false;
  bool lastPartWasRanked = false;
};
} // namespace

template<typename T>
Fortran::lower::MaterializedExpr Fortran::lower::DesignatorBuilder<T>::gen(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Designator<T>& designator, Fortran::lower::ExplicitIterSpace* explicitIterSpace, bool loadArrays) {
  return DesignatorBuilderImpl(loc, converter, stmtCtx, explicitIterSpace, loadArrays).genDesignator(designator);
}

// Ensure DesignatorBuilder is instantiated for all the required types.
namespace Fortran {
using namespace evaluate;
FOR_EACH_INTRINSIC_KIND(template struct Fortran::lower::DesignatorBuilder, )
template struct lower::DesignatorBuilder<SomeDerived>;
}

//===--------------------------------------------------------------------===//
// MaterializedExpr utilities implementations
//===--------------------------------------------------------------------===//

void Fortran::lower::MaterializedExpr::prepareForAddressing(fir::FirOpBuilder& builder, mlir::Location loc, bool loadArrays) {
  if (readyForAddressing || baseIsArrayLoad())
    return;
  const auto& exv = std::get<fir::ExtendedValue>(loweredBase);  
  if (const auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>())
    loweredBase = fir::factory::genMutableBoxRead(builder, loc, *mutableBox);

  if (isArray()) {
    shape = createShape(builder, loc);
    slice = createSlice(builder, loc);
    if (loadArrays)
      TODO(loc, "array load");
  }
  readyForAddressing = true;
}

bool Fortran::lower::MaterializedExpr::isArray() const {
  if (!loweredSubscripts.empty())
    return true;
  return std::visit(common::visitors{[&](const fir::ExtendedValue& base) -> bool {
    return base.rank() != 0;
  },
  [&](const fir::ArrayLoadOp& load) -> bool {
    // If the array load base rank is not "consumed" by indexes, then
    // this is an array.
    return preRankedPath.empty() || preRankedPath[0].getType().isa<fir::FieldType>();
  }}, loweredBase);
}


mlir::Value
Fortran::lower::MaterializedExpr::createShape(fir::FirOpBuilder &builder,
                                                mlir::Location loc) const {
  assert(!baseIsArrayLoad() && "shape already created");
  return builder.createShape(loc, std::get<fir::ExtendedValue>(loweredBase));
}

static mlir::Value getLengthFromComponentPath(fir::FirOpBuilder &builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> componentPath) {
  fir::FieldIndexOp lastField;
  for (auto component : llvm::reverse(componentPath)) {
    if (auto field = component.getDefiningOp<fir::FieldIndexOp>()) {
      lastField = field;
      break;
    }
  }
  if (!lastField)
    fir::emitFatalError(loc, "expected component reference in designator");
  auto recTy = lastField.on_type().cast<fir::RecordType>(); 
  auto charType = recTy.getType(lastField.field_id()).cast<fir::CharacterType>();
  // Derived type components with non constant length are F2003.
  if (charType.hasDynamicLen())
    TODO(loc, "designator with derived type length parameters");
  return builder.createIntegerConstant(loc, builder.getCharacterLengthType(), charType.getLen());
}

llvm::SmallVector<mlir::Value>
Fortran::lower::MaterializedExpr::getTypeParams(fir::FirOpBuilder &builder, mlir::Location loc) const {
  if (baseIsArrayLoad())
    TODO(loc, "arrayLoad params");

  const auto& baseExv = std::get<fir::ExtendedValue>(loweredBase);
  if (isExtendedValue())
    return fir::factory::getTypeParams(builder, loc, baseExv);
  
  auto elementType = getElementType();
  if (elementType.isa<fir::CharacterType>()) {
    mlir::Value len = componentPath.empty() ?
      fir::factory::readCharLen(builder, loc, baseExv) :
      getLengthFromComponentPath(builder, loc, componentPath);
    if (substringBounds.empty())
      return {len};
    auto upper = substringBounds.size() == 2 ? substringBounds[1] : len;
    auto charLenType = builder.getCharacterLengthType();
    upper = builder.createConvert(loc, charLenType, upper); 
    auto lower = builder.createConvert(loc, charLenType, substringBounds[0]); 
    auto zero = builder.createIntegerConstant(loc, charLenType, 0);
    auto one = builder.createIntegerConstant(loc, charLenType, 1);
    auto diff = builder.create<mlir::arith::SubIOp>(loc, upper, lower);
    auto newLen = builder.create<mlir::arith::AddIOp>(loc, diff, one);
    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle, lower, upper);
    auto select = builder.create<mlir::SelectOp>(loc, cmp, newLen, zero);
    return {select.getResult()};
  }
  if (auto recordType = elementType.dyn_cast<fir::RecordType>())
    if (recordType.getNumLenParams() != 0)
      TODO(loc, "derived type designator with length parameters");
  return {};
}

mlir::Value
Fortran::lower::MaterializedExpr::createSlice(fir::FirOpBuilder &builder,
                                                mlir::Location loc) const {
  // FIXME: does this work with array%comp ?
  if (loweredSubscripts.empty())
    return {};
  auto idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> triples;
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
  for (const auto &subscript : loweredSubscripts)
    std::visit(Fortran::common::visitors{
                   [&](const LoweredTriplet &triplet) {
                     triples.emplace_back(triplet.lb);
                     triples.emplace_back(triplet.ub);
                     triples.emplace_back(triplet.stride);
                   },
                   [&](const LoweredVectorSubscript &vector) {
                     triples.emplace_back(one);
                     triples.emplace_back(vector.size);
                     triples.emplace_back(one);
                   },
                   [&](const mlir::Value &i) {
                     triples.emplace_back(i);
                     triples.emplace_back(undef);
                     triples.emplace_back(undef);
                   },
               },
               subscript);
  return builder.create<fir::SliceOp>(loc, triples, componentPath);
}


mlir::Type Fortran::lower::MaterializedExpr::getBaseType() const {
  auto elementType = getElementType();
  if (!isArray())
    return elementType;

  if (!loweredSubscripts.empty()) {
    auto rank = 0;
    for (const auto &subscript : loweredSubscripts)
      if (!std::holds_alternative<mlir::Value>(subscript))
        ++rank;
    auto unknownExtent = fir::SequenceType::getUnknownExtent();
    auto shape = fir::SequenceType::Shape(rank, unknownExtent);
    return fir::SequenceType::get(shape, elementType);
  }

  return std::visit(Fortran::common::visitors{
    [&](const fir::ExtendedValue& baseExv) -> mlir::Type {
      auto baseSeqType = fir::unwrapPassByRefType(fir::getBase(baseExv).getType());
      auto shape = baseSeqType.cast<fir::SequenceType>().getShape();
      return fir::SequenceType::get(shape, elementType);
    },
    [&](const fir::ArrayLoadOp& arrayLd) -> mlir::Type {
      fir::ArrayLoadOp arrayLoad = arrayLd;
      if (preRankedPath.empty())
        return arrayLoad.getType();
      if (auto field = preRankedPath.back().getDefiningOp<fir::FieldIndexOp>()) {
        auto lastFieldType =  field.on_type().cast<fir::RecordType>().getType(field.field_id());
        return lastFieldType;
      }
      return elementType;
    },
  }
  , loweredBase);
}

llvm::SmallVector<mlir::Value>
Fortran::lower::MaterializedExpr::getExtents(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) const {
  if (!isArray()) 
    return {};

  if (slice)
    return computeSliceShape(builder, loc, slice);
  if (shape && !shape.getType().isa<fir::ShiftType>()) {
    auto extents = fir::factory::getExtents(shape);
    return {extents.begin(), extents.end()};
  }

  if (loweredSubscripts.empty()) {
    if (baseIsArrayLoad()) {
      auto arrayLoadShape = getShape(builder, loc, std::get<fir::ArrayLoadOp>(loweredBase));
      if (preRankedPath.empty() || preRankedPath[0].getType().isa<fir::FieldType>())
        return arrayLoadShape;
      // Otherwise, preRankedPath indexes consumed the arrayload shape.
      return {}; 
    }
    const auto& baseExv = std::get<fir::ExtendedValue>(loweredBase);
    return fir::factory::getExtents(builder, loc, baseExv);
  }
  
  auto idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;
  for (const auto &subscript : loweredSubscripts) {
    if (std::holds_alternative<mlir::Value>(subscript))
      continue;
    if (const auto *triplet = std::get_if<LoweredTriplet>(&subscript)) {
      auto ext = builder.genExtentFromTriplet(loc, triplet->lb, triplet->ub,
                                                 triplet->stride, idxTy);
      extents.push_back(ext);
    } else {
      const auto &vector = std::get<LoweredVectorSubscript>(subscript);
      extents.push_back(vector.size);
    }
  }
  return extents;
}

fir::ExtendedValue Fortran::lower::MaterializedExpr::getElementAt(
    fir::FirOpBuilder &builder, mlir::Location loc,
    mlir::ValueRange indices) const {
  if (baseIsArrayLoad())
    TODO(loc, "element at array load");
  const auto& baseExv = std::get<fir::ExtendedValue>(loweredBase);
  auto element = isArray() ? genArrayCoor(builder, loc, indices) : baseExv;
  if (!substringBounds.empty()) {
    auto *charBox = element.getCharBox();
    assert(charBox && "substring requires CharBox base");
    fir::factory::CharacterExprHelper helper{builder, loc};
    return helper.createSubstring(*charBox, substringBounds);
  }
  return element;
}

std::pair<llvm::SmallVector<mlir::Value>, bool> Fortran::lower::MaterializedExpr::genPath(fir::FirOpBuilder& builder, mlir::Location loc, mlir::ValueRange indices) const {
  llvm::SmallVector<mlir::Value> path;
  bool flagUserOffsets = false;
  path.append(preRankedPath.begin(), preRankedPath.end());
  auto* arrayLoad = std::get_if<fir::ArrayLoadOp>(&loweredBase);
  if (arrayLoad) {
    fir::ArrayLoadOp arrayLd = *arrayLoad;
    if (!arrayLd.slice() && !loweredSubscripts.empty()) {
    // Slice was not created on array load because the array_load was created
    // before the slice op could be created (happens in Forall contexts).
    auto idxTy = builder.getIndexType();
    auto indicePosition = 0;
    for (const auto &subscript : loweredSubscripts)
      std::visit(Fortran::common::visitors{
                     [&](const LoweredTriplet &triplet) {
                       TODO(loc, "triplet in raised forall"); 
                       indicePosition++;
                     },
                     [&](const LoweredVectorSubscript &vector) {
                        auto idx = indices[indicePosition];
                       auto vecElt = vector.getVector().getElementValueAt(builder, loc, {idx});
                       path.emplace_back(
                           builder.createConvert(loc, idxTy, fir::getBase(vecElt)));
                       indicePosition++;
                     },
                     [&](const mlir::Value &i) {
                       path.emplace_back(builder.createConvert(loc, idxTy, i));
                     },
                 },
                 subscript);
    // Flag the offsets as "Fortran" as they are not zero-origin.
    flagUserOffsets = true;
    }
  } else {
    path.append(indices.begin(), indices.end());
  }
  path.append(componentPath.begin(), componentPath.end());
  return {path, flagUserOffsets};
}

mlir::Value Fortran::lower::MaterializedExpr::genArrayFetchAt(
    fir::FirOpBuilder &builder, mlir::Location loc,
    mlir::ValueRange indices) const {
  assert(baseIsArrayLoad() && "array must be loaded to use fir.array_update");
  auto arrayLoad = std::get<fir::ArrayLoadOp>(loweredBase);
  auto [path, flagUserOffsets] = genPath(builder, loc, indices);
  auto fetch = builder.create<fir::ArrayFetchOp>(
    loc, elementType, arrayLoad, path,
    /*FIXME:*/llvm::None);
  if (flagUserOffsets)
    fetch->setAttr(fir::factory::attrFortranArrayOffsets(),
                    builder.getUnitAttr());
  return fetch;
}

mlir::Value Fortran::lower::MaterializedExpr::genArrayUpdateAt(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value newElementValue,
    mlir::ValueRange indices, mlir::Value previousArrayValue) const {
  assert(previousArrayValue && baseIsArrayLoad() && "array must be loaded to use fir.array_update");
  auto arrTy = previousArrayValue.getType();
  auto [path, flagUserOffsets] = genPath(builder, loc, indices);
  auto newValue = builder.createConvert(loc, fir::unwrapSequenceType(arrTy), newElementValue);
  auto update = builder.create<fir::ArrayUpdateOp>(
    loc, arrTy, previousArrayValue, newValue, path,
    /*FIXME:*/llvm::None);
  if (flagUserOffsets)
    update->setAttr(fir::factory::attrFortranArrayOffsets(),
                    builder.getUnitAttr());
  return update;
}


fir::ExtendedValue Fortran::lower::MaterializedExpr::genArrayCoor(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::ValueRange indices) const {
  /// Generate the indexes for the array_coor inside the loops.
  llvm::SmallVector<mlir::Value> inductionVariables;
  assert(!baseIsArrayLoad() && "array_coor is not intended for loaded arrays");
  const auto& baseExv = std::get<fir::ExtendedValue>(loweredBase);
  auto memrefTy = fir::getBase(baseExv).getType();
  auto idx = fir::factory::originateIndices(loc, builder, memrefTy, shape, indices);
  inductionVariables.append(idx.begin(), idx.end());
  auto idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> indexes;
  if (!loweredSubscripts.empty()) {
    auto inductionIdx = inductionVariables.size() - 1;
    for (const auto &subscript : loweredSubscripts)
      std::visit(Fortran::common::visitors{
                     [&](const LoweredTriplet &triplet) {
                       indexes.emplace_back(inductionVariables[inductionIdx--]);
                     },
                     [&](const LoweredVectorSubscript &vector) {
                       auto vecIndex = inductionVariables[inductionIdx--];
                       auto vecEltRef = vector.getVector().getElementAt(builder, loc, {vecIndex});
                       auto vecElt = builder.create<fir::LoadOp>(loc, fir::getBase(vecEltRef));
                       indexes.emplace_back(
                           builder.createConvert(loc, idxTy, vecElt));
                     },
                     [&](const mlir::Value &i) {
                       indexes.emplace_back(builder.createConvert(loc, idxTy, i));
                     },
                 },
                 subscript);
  } else {
    for (auto index : llvm::reverse(inductionVariables))
      indexes.push_back(index);
  }
  auto refTy = builder.getRefType(getElementType());
  auto elementAddr = builder.create<fir::ArrayCoorOp>(
      loc, refTy, fir::getBase(baseExv), shape, slice, indexes,
      fir::getTypeParams(baseExv));
  return fir::factory::arraySectionElementToExtendedValue(
      builder, loc, baseExv, elementAddr, slice);
}

bool Fortran::lower::MaterializedExpr::hasVectorSubscripts() const {
  for (const auto &subscript : loweredSubscripts)
    if (std::holds_alternative<LoweredVectorSubscript>(subscript))
      return true;
  return false;
}

fir::ExtendedValue Fortran::lower::MaterializedExpr::asBox(fir::FirOpBuilder& builder, mlir::Location loc) const {
  assert(!baseIsArrayLoad() && "cannot box array_load");
  assert(!hasVectorSubscripts() && "cannot box vector subscripted designator");
  const auto& baseExv = std::get<fir::ExtendedValue>(loweredBase);
  auto memref = fir::getBase(baseExv);
  auto boxTy = fir::BoxType::get(getBaseType());
  auto shape = createShape(builder, loc);
  auto slice = createSlice(builder, loc);
  if (memref.getType().isa<fir::BoxType>())
    return builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice);
  return builder.create<fir::EmboxOp>(loc, boxTy, memref, shape, slice, fir::getTypeParams(baseExv));
}

fir::ExtendedValue Fortran::lower::MaterializedExpr::getAsExtendedValue(fir::FirOpBuilder& builder, mlir::Location loc) const {
  assert(!baseIsArrayLoad() && "cannot get array_load as ExtendedValue");
  if (isExtendedValue())
    return std::get<fir::ExtendedValue>(loweredBase);
  return asBox(builder, loc);
}

llvm::SmallVector<mlir::Value> Fortran::lower::MaterializedExpr::getLBounds(fir::FirOpBuilder &builder, mlir::Location loc) const {
  if (shape && !slice) {
    auto origins = fir::factory::getOrigins(shape);
    return {origins.begin(), origins.end()};
  }

  if (baseIsArrayLoad())
    TODO(loc, "lower bounds of array loaded base");

  // Array sections do not have default lower bounds.
  if (!isExtendedValue())
    return {};

  auto lbs = std::get<fir::ExtendedValue>(loweredBase).match(
    [&](fir::CharArrayBoxValue& array) -> llvm::ArrayRef<mlir::Value> {
      return array.getLBounds();
    },
    [&](fir::ArrayBoxValue& array) -> llvm::ArrayRef<mlir::Value> {
      return array.getLBounds();
    },
    [&](fir::BoxValue& array) -> llvm::ArrayRef<mlir::Value> {
      return array.getLBounds();
    },
    [&](auto&) -> llvm::ArrayRef<mlir::Value> {
      // TODO: mutableBox ?
      return {};
    }
  );
  return {lbs.begin(), lbs.end()};
}

Fortran::lower::MaterializedExpr::LoweredVectorSubscript::LoweredVectorSubscript(const LoweredVectorSubscript&) = default;
Fortran::lower::MaterializedExpr::LoweredVectorSubscript::LoweredVectorSubscript(LoweredVectorSubscript&&) = default;
Fortran::lower::MaterializedExpr::LoweredVectorSubscript& Fortran::lower::MaterializedExpr::LoweredVectorSubscript::operator=(const LoweredVectorSubscript&) = default;
Fortran::lower::MaterializedExpr::LoweredVectorSubscript& Fortran::lower::MaterializedExpr::LoweredVectorSubscript::operator=(LoweredVectorSubscript&&) = default;
Fortran::lower::MaterializedExpr::LoweredVectorSubscript::~LoweredVectorSubscript() = default;

Fortran::lower::MaterializedExpr::LoweredVectorSubscript::LoweredVectorSubscript(Fortran::lower::LoweredExpr&& expr, mlir::Value size) : vector{std::move(expr)}, size{size} {}

const Fortran::lower::LoweredExpr& Fortran::lower::MaterializedExpr::LoweredVectorSubscript::getVector() const {
  return vector.value();
}
Fortran::lower::LoweredExpr& Fortran::lower::MaterializedExpr::LoweredVectorSubscript::getVector() {
  return vector.value();
}
