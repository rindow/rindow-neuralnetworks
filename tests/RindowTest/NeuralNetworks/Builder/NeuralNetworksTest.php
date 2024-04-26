<?php
namespace RindowTest\NeuralNetworks\Builder\NeuralNetworksTest;

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend as RindowBlasBackend;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend as RindowCLBlastBackend;


use PHPUnit\Framework\TestCase;

class NeuralNetworksTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function testconstructor()
    {
        $mo = $this->newMatrixOperator();
        $nn = new NeuralNetworks($mo);

        if($nn->backend()->accelerated()) {
            $this->assertInstanceof(RindowCLBlastBackend::class,$nn->backend());
        } else {
            $this->assertInstanceof(RindowBlasBackend::class,$nn->backend());
        }
    }

    public function testDeviceType()
    {
        $mo = $this->newMatrixOperator();
        $nn = new NeuralNetworks($mo);

        if($nn->backend()->accelerated()) {
            $deviceType = implode(',',$nn->backend()->primaryLA()->deviceTypes());
        } else {
            $serviceLevel = $mo->la()->service()->serviceLevel();
            if($serviceLevel>1) {
                $deviceType = 'CPU';
            } else {
                $deviceType = 'PHP';
            }
        }
        $this->assertEquals($deviceType,$nn->deviceType());
    }

}
