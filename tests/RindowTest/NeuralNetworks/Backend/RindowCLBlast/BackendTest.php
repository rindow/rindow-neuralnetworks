<?php
namespace RindowTest\NeuralNetworks\Backend\RindowCLBlast\BackendTest;

//if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
//    require_once __DIR__.'/../../../../../../rindow-math-matrix/tests/RindowTest/Math/Matrix/LinearAlgebraTest.php';
//}
if(!class_exists('RindowTest\NeuralNetworks\Backend\RindowBlas\BackendTest\Test')) {
    require_once __DIR__.'/../RindowBlas/BackendTest.php';
}

use RindowTest\NeuralNetworks\Backend\RindowBlas\BackendTest\Test as ORGTest;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend;
use Rindow\Math\Matrix\MatrixOperator;

/**
*   @requires extension rindow_clblast
*/
class Test extends ORGTest
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newBackend($mo)
    {
        return new Backend($mo);
    }
}
