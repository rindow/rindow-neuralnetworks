<?php
namespace RindowTest\NeuralNetworks\Backend\RindowCLBlast\BackendTest;

//if(!class_exists('RindowTest\Math\Matrix\LinearAlgebraTest\Test')) {
//    require_once __DIR__.'/../../../../../../rindow-math-matrix/tests/RindowTest/Math/Matrix/LinearAlgebraTest.php';
//}
if(!class_exists('RindowTest\NeuralNetworks\Backend\RindowBlas\BackendTest\BackendTest')) {
    require_once __DIR__.'/../RindowBlas/BackendTest.php';
}

use RindowTest\NeuralNetworks\Backend\RindowBlas\BackendTest\BackendTest as ORGTest;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend;
use Rindow\Math\Matrix\MatrixOperator;

class BackendTest extends ORGTest
{
    public function setUp() : void
    {
        parent::setUp();
        $mo = new MatrixOperator();
        if(!$mo->isAccelerated()) {
            $this->markTestSkipped("The service is not Accelerated.");
            return;
        }
    }

    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newBackend($mo)
    {
        return new Backend($mo);
    }
}
