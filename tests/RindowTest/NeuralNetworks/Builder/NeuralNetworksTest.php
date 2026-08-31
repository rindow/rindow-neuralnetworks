<?php
namespace RindowTest\NeuralNetworks\Builder\NeuralNetworksTest;

use Rindow\Math\Matrix\Drivers\Service;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend as RindowBlasBackend;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend as RindowCLBlastBackend;
use Interop\Polite\Math\Matrix\DeviceBuffer;
use Interop\Polite\Math\Matrix\LinearBuffer;

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
            if($serviceLevel>=Service::LV_ADVANCED) {
                $deviceType = 'CPU';
            } else {
                $deviceType = 'PHP';
            }
        }
        $this->assertEquals($deviceType,$nn->deviceType());
    }

    public function testLA()
    {
        $mo = $this->newMatrixOperator();
        $nn = new NeuralNetworks($mo);

        if($nn->backend()->accelerated()) {
            $this->assertInstanceof(\Rindow\Math\Matrix\LinearAlgebraCL::class,$nn->la());
        } else {
            $this->assertInstanceof(\Rindow\Math\Matrix\LinearAlgebra::class,$nn->la());
        }
    }

    public function testDeviceArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = new NeuralNetworks($mo);
        $serviceLevel = $mo->la()->service()->serviceLevel();
        if($serviceLevel<Service::LV_ADVANCED) {
            $this->markTestSkipped("The service is not Accelerated.");
            return;
        }

        $a = $nn->deviceArray($mo->array([1,2,3])); // host array to device array
        $buffer = $a->buffer();
        if($nn->backend()->accelerated()) {
            $this->assertInstanceOf(DeviceBuffer::class,$buffer);
        } else {
            $this->assertInstanceOf(LinearBuffer::class,$buffer);
        }

        $b = $nn->deviceArray($a); // device array to device array
        $buffer = $b->buffer();
        if($nn->backend()->accelerated()) {
            $this->assertInstanceOf(DeviceBuffer::class,$buffer);
        } else {
            $this->assertInstanceOf(LinearBuffer::class,$buffer);
        }
    }

    public function testHostArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = new NeuralNetworks($mo);
        $serviceLevel = $mo->la()->service()->serviceLevel();
        if($serviceLevel<Service::LV_ADVANCED) {
            $this->markTestSkipped("The service is not Accelerated.");
            return;
        }

        $a = $nn->deviceArray($mo->array([1,2,3])); // host array to device array
        $a = $nn->hostArray($a);  // device array to host array
        $buffer = $a->buffer();
        $this->assertInstanceOf(LinearBuffer::class,$buffer);
        $a = $nn->hostArray($a);  // host array to host array
        $buffer = $a->buffer();
        $this->assertInstanceOf(LinearBuffer::class,$buffer);
    }

}
