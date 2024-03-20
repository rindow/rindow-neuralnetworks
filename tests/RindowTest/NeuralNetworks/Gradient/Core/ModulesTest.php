<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\ModulesTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Modules;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Layer\Flatten;
use Rindow\NeuralNetworks\Layer\Dropout;
use Interop\Polite\Math\Matrix\Buffer;

class ModulesTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function newBackend($nn)
    {
        return $nn->backend();
    }

    
    public function testCreateAndCount()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $shape = new Modules();
        $class = new class ($nn) extends AbstractModel {
            protected $l;
            public function __construct($nn) {
                parent::__construct($nn);
                $this->l = $nn->gradient()->Modules([
                    $nn->layers->Dense(2),
                    $nn->layers->Dense(2),
                    $nn->layers->Dense(2),
                ]);
            }

            public function call($inputs)
            {
                return $inputs;
            }
        };

        $subs = $class->submodules();
        $this->assertCount(1,$subs);
        $this->assertCount(3,$subs[0]);
    }

    public function testArrayAccess()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $modules = $g->Modules([
            $nn->layers->Dense(2),
            $nn->layers->Flatten(),
            $nn->layers->Dropout(0.5),
        ]);
        $this->assertInstanceof(Dense::class,$modules[0]);
        $this->assertInstanceof(Flatten::class,$modules[1]);
        $this->assertInstanceof(Dropout::class,$modules[2]);
    }

    public function testAdd()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $modules = $g->Modules();
        $modules->add($nn->layers->Dense(2));
        $modules->add($nn->layers->Flatten());
        $modules->add($nn->layers->Dropout(0.5));
        
        $this->assertInstanceof(Dense::class,$modules[0]);
        $this->assertInstanceof(Flatten::class,$modules[1]);
        $this->assertInstanceof(Dropout::class,$modules[2]);
    }

    public function testIterableAccess()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $mdls = $g->Modules([
            $nn->layers->Dense(2),
            $nn->layers->Flatten(),
            $nn->layers->Dropout(0.5),
        ]);
        $modules = [];
        $idx = 0;
        foreach($mdls as $i => $m) {
            $this->assertEquals($idx,$i);
            $idx ++;
            $modules[] = $m;
        }
        $this->assertInstanceof(Dense::class,$modules[0]);
        $this->assertInstanceof(Flatten::class,$modules[1]);
        $this->assertInstanceof(Dropout::class,$modules[2]);
    }
}