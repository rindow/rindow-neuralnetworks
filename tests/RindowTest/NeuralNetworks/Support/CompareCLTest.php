<?php
namespace RindowTest\NeuralNetworks\Support\CompareCLTest;

use PHPUnit\Framework\TestCase;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend as BackendCL;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Activation\FunctionFactory;
use Interop\Polite\Math\Matrix\NDArray;

/**
*   @requires extension rindow_clblast
*/
class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newBackend($mo)
    {
        return new Backend($mo);
    }

    public function newBackendCL($mo)
    {
        return new BackendCL($mo);
    }

    public function testDenseWeightsImmediate()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $g = $nn->gradient();
        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);
        $gcl = $nncl->gradient();

        $dense = $nn->layers()->Dense($units=2);
        $densecl = $nncl->layers()->Dense($units=2);
        $lossfunc = $nn->losses->SparseCategoricalCrossEntropy(from_logits:true);
        $lossfunccl = $nncl->losses->SparseCategoricalCrossEntropy(from_logits:true);
        $optimizer = $nn->optimizers->Adam();
        $optimizercl = $nncl->optimizers->Adam();
        $outputs = $dense->forward($g->Variable($backend->zeros([1,2])),true);
        $outputscl = $densecl->forward($gcl->Variable($backendCL->zeros([1,2])),true);

        $weights = $dense->trainableVariables();
        $weightsCL = $densecl->trainableVariables();
        
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $backendCL->copy($backendCL->array($w->value()),$wcl->value());
        }
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }

        $inputs = $g->Variable($backend->array([[1, 3],]));
        $inputscl = $gcl->Variable($backendCL->array([[1, 3],]));
        $trues = $g->Variable($backend->array([0,]));
        $truescl = $gcl->Variable($backendCL->array([0,]));

        $outputs = $dense->forward($inputs,true);
        $outputscl = $densecl->forward($inputscl,true);
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }

        $this->assertEquals($outputs->toArray(),$outputscl->toArray());

        $loss = $nn->with($tape=$g->GradientTape(),function() use ($dense,$lossfunc,$inputs,$trues) {
            $outputs = $dense->forward($inputs,true);
            $loss = $lossfunc->forward($trues,$outputs);
            return $loss;
        });
        $params = $dense->trainableVariables();
        $grads = $tape->gradient($loss,$params);

        $losscl = $nncl->with($tapecl=$gcl->GradientTape(),function() use ($densecl,$lossfunccl,$inputscl,$truescl) {
            $outputs = $densecl->forward($inputscl,true);
            $loss = $lossfunccl->forward($truescl,$outputs);
            return $loss;
        });
        $paramscl = $densecl->trainableVariables();
        $gradscl = $tapecl->gradient($losscl,$paramscl);

        foreach (array_map(null,$grads,$gradscl) as [$grad,$gradcl]) {
            $diff = $backend->sub($grad,$backendCL->ndarray($gradcl));
            $this->assertLessThan(1e-6,$backend->scalar($backend->amax($diff)));
        }

        $optimizer->update($params,$grads);
        $optimizercl->update($paramscl,$gradscl);

        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $diff = $backend->sub($w,$backendCL->ndarray($wcl));
            $this->assertLessThan(1e-6,$backend->scalar($backend->amax($diff)));
        }
    }

    public function testDenseWeightsGraphFunc()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $g = $nn->gradient();
        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);
        $gcl = $nncl->gradient();

        $dense = $nn->layers()->Dense($units=2);
        $densecl = $nncl->layers()->Dense($units=2);
        $lossfunc = $nn->losses->SparseCategoricalCrossEntropy(from_logits:true);
        $lossfunccl = $nncl->losses->SparseCategoricalCrossEntropy(from_logits:true);
        $optimizer = $nn->optimizers->Adam();
        $optimizercl = $nncl->optimizers->Adam();
        $func = $nn->gradient->Function(function($inputs,$trues) use ($dense) {
            $outputs = $dense->forward($inputs,true);
            return $outputs;
        });
        $funccl = $nncl->gradient->Function(function($inputs,$trues) use ($densecl) {
            $outputs = $densecl->forward($inputs,true);
            return $outputs;
        });

        $inputs = $g->Variable($backend->array([[1, 3],]));
        $inputscl = $gcl->Variable($backendCL->array([[1, 3],]));
        $trues = $g->Variable($backend->array([0,]));
        $truescl = $gcl->Variable($backendCL->array([0,]));

        $outputs = $func($inputs,$trues);
        $outputscl = $funccl($inputscl,$truescl);

        $weights = $dense->trainableVariables();
        $weightsCL = $densecl->trainableVariables();
        
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $backendCL->copy($backendCL->array($w->value()),$wcl->value());
        }
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }

        $outputs = $func($inputs,$trues);
        $outputscl = $funccl($inputscl,$trues);
        $this->assertEquals($outputs->toArray(),$outputscl->toArray());

        $loss = $nn->with($tape=$g->GradientTape(),function() use ($func,$lossfunc,$inputs,$trues) {
            $outputs = $func($inputs,$trues);
            $loss = $lossfunc->forward($trues,$outputs);
            return $loss;
        });
        $params = $dense->trainableVariables();
        $grads = $tape->gradient($loss,$params);

        $losscl = $nncl->with($tapecl=$gcl->GradientTape(),function() use ($funccl,$lossfunccl,$inputscl,$truescl) {
            $outputs = $funccl($inputscl,$truescl);
            $loss = $lossfunccl->forward($truescl,$outputs);
            return $loss;
        });
        $paramscl = $densecl->trainableVariables();
        $gradscl = $tapecl->gradient($losscl,$paramscl);

        foreach (array_map(null,$grads,$gradscl) as [$grad,$gradcl]) {
            $diff = $backend->sub($grad,$backendCL->ndarray($gradcl));
            $this->assertLessThan(1e-6,$backend->scalar($backend->amax($diff)));
        }

        $optimizer->update($params,$grads);
        $optimizercl->update($paramscl,$gradscl);

        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $diff = $backend->sub($w,$backendCL->ndarray($wcl));
            $this->assertLessThan(1e-6,$backend->scalar($backend->amax($diff)));
        }
    }

    public function testFitDense()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=32,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,
                activation:'softmax'),
            //$nn->layers()->Dense($units=2),
        ]);
        $model->compile(loss:$nn->losses->SparseCategoricalCrossEntropy(from_logits:true));

        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);

        $modelcl = $nncl->models()->Sequential([
            $nncl->layers()->Dense($units=32,input_shape:[2],
                activation:'sigmoid'),
            $nncl->layers()->Dense($units=2,
                activation:'softmax'),
            //$nncl->layers()->Dense($units=2),
        ]);
        $modelcl->compile(loss:$nncl->losses->SparseCategoricalCrossEntropy(from_logits:true));

        $g = $nn->gradient();
        $gcl = $nncl->gradient();
        $model->forward($g->Variable($backend->zeros([1,2])));
        $modelcl->forward($gcl->Variable($backendCL->zeros([1,2])));

        $weights = $model->trainableVariables();
        $weightsCL = $modelcl->trainableVariables();
        
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $backendCL->copy($backendCL->array($w->value()),$wcl->value());
        }
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }

        $optimizer = $model->optimizer();
        $optimizer->build($weights);
        $oweights = $optimizer->getWeights();
        $optimizerCL = $modelcl->optimizer();
        $optimizerCL->build($weightsCL);
        $oweightsCL = $optimizerCL->getWeights();
        foreach (array_map(null,$oweights,$oweightsCL) as [$w,$wcl]) {
            $backendCL->copy($backendCL->array($w),$wcl);
        }
        foreach (array_map(null,$oweights,$oweightsCL) as [$w,$wcl]) {
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }

        // training greater or less
        $x = $mo->array([[1, 3],]);
        $t = $mo->array([0,]);
        $history = $model->fit($x,$t,epochs:1,verbose:0);
        $history = $modelcl->fit($x,$t,epochs:1,verbose:0);

        $weights = $model->trainableVariables();
        $weightsCL = $modelcl->trainableVariables();
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $diff = $backend->sub($w->value(),$backendCL->ndarray($wcl));
            //echo "[".implode(',',$w->shape())."]";
            //echo $mo->toString($diff,'%5.3e')."\n";
            //if($w->toArray()!=$wcl->toArray()) {
            //    echo "w=".$mo->toString($w,'%5.3e')."\n";
            //    echo "wcl=".$mo->toString($backendCL->ndarray($wcl),'%5.3e')."\n";
            //}
            $this->assertLessThan(1e-6,$backend->scalar($backend->amax($diff)));
        }
    }

    public function testFitConv2D()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $inputShape = [28,28,1];

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv2D(
               $filters=32,
                $kernel_size=3,
                input_shape:$inputShape,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->Conv2D(
               $filters=32,
                $kernel_size=3,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->MaxPooling2D(),
            $nn->layers()->Conv2D(
               $filters=64,
                $kernel_size=3,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->Conv2D(
               $filters=64,
                $kernel_size=3,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->MaxPooling2D(),
            $nn->layers()->Flatten(),
            $nn->layers()->Dense($units=512,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->Dense($units=10,
                activation:'softmax'),
        ]);
        $model->compile();

        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);

        $modelcl = $nncl->models()->Sequential([
            $nncl->layers()->Conv2D(
               $filters=32,
                $kernel_size=3,
                input_shape:$inputShape,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nncl->layers()->Conv2D(
               $filters=32,
                $kernel_size=3,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nncl->layers()->MaxPooling2D(),
            $nncl->layers()->Conv2D(
               $filters=64,
                $kernel_size=3,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nncl->layers()->Conv2D(
               $filters=64,
                $kernel_size=3,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nncl->layers()->MaxPooling2D(),
            $nncl->layers()->Flatten(),
            $nncl->layers()->Dense($units=512,
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nncl->layers()->Dense($units=10,
                activation:'softmax'),
        ]);
        $modelcl->compile();

        $g = $nn->gradient();
        $gcl = $nncl->gradient();
        $model->forward($g->Variable($backend->zeros(array_merge([1],$inputShape))));
        $modelcl->forward($gcl->Variable($backendCL->zeros(array_merge([1],$inputShape))));

        $weights = $model->trainableVariables();
        $weightsCL = $modelcl->trainableVariables();
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $backendCL->copy($backendCL->array($w->value()),$wcl->value());
        }
        foreach (array_map(null,$weights,$weightsCL) as [$w,$wcl]) {
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }

        $optimizer = $model->optimizer();
        $optimizer->build($weights);
        $oweights = $optimizer->getWeights();
        $optimizerCL = $modelcl->optimizer();
        $optimizerCL->build($weightsCL);
        $oweightsCL = $optimizerCL->getWeights();
        foreach ($oweights as $key => $w) {
            $wcl = $oweightsCL[$key];
            $backendCL->copy($backendCL->array($w),$wcl);
        }

        foreach ($weights as $key => $w) {
            $wcl = $weightsCL[$key];
            $this->assertEquals($w->toArray(),$wcl->toArray());
        }
        // training greater or less
        $x = $backend->glorot_normal(array_merge([32],$inputShape));
        $t = $mo->la()->randomUniform([32],0,9,NDArray::int32);
        $history = $model->fit($x,$t,epochs:1,verbose:0);
        $history = $modelcl->fit($x,$t,epochs:1,verbose:0);

        foreach ($weights as $key => $w) {
            $wcl = $weightsCL[$key];
            $diff = $backend->sub($w,$backendCL->ndarray($wcl));
            //echo "[".implode(',',$w->shape())."]";
            //echo $mo->toString($diff,'%5.3e')."\n";
            //echo sprintf('%5.3e',$backend->asum($diff)).",";
            //if($w->toArray()!=$wcl->toArray()) {
            //    echo "w=".$mo->toString($w,'%5.3e')."\n";
            //    echo "wcl=".$mo->toString($backendCL->ndarray($wcl),'%5.3e')."\n";
            //}
            $this->assertLessThan(1e-5,$backend->scalar($backend->amax($diff)));
        }
    }

    public function testDenseLayerWithSigmoid()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);

        $dense = $nn->layers()->Dense($units=32,input_shape:[2],
            activation:'sigmoid');
        $dense->build();
        $densecl = $nncl->layers()->Dense($units=32,input_shape:[2],
            activation:'sigmoid');
        $densecl->build();

        $weights = $dense->getParams();
        $weightsCL = $densecl->getParams();
        foreach ($weights as $key => $w) {
            $wcl = $weightsCL[$key];
            $backendCL->copy($backendCL->array($w),$wcl);
        }

        $x = $backend->array([[1, 3],]);
        $y = $dense->forward($x,true);
        $xcl = $backendCL->array([[1, 3],]);
        $ycl = $densecl->forward($xcl,true);
        //echo $mo->toString($backend->sub($y,$backendCL->ndarray($ycl)));
        $diff = $backend->sub($y,$backendCL->ndarray($ycl));
        $this->assertLessThan(1e-7,$backend->scalar($backend->amax($diff)));
    }

    public function testBatchGemm()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);

        $a = $backend->glorot_normal([32,2]);
        $b = $backend->glorot_normal([2,32]);
        $c = $backend->glorot_normal([32]);
        $acl = $backendCL->zerosLike($a);
        $bcl = $backendCL->zerosLike($b);
        $ccl = $backendCL->zerosLike($c);
        $backendCL->copy($backendCL->array($a),$acl);
        $backendCL->copy($backendCL->array($b),$bcl);
        $backendCL->copy($backendCL->array($c),$ccl);
        $o = $backend->batch_gemm($a,$b,1.0,1.0,$c);
        $ocl = $backendCL->batch_gemm($acl,$bcl,1.0,1.0,$ccl);
        $diff = $backend->sub($o,$backendCL->ndarray($ocl));
        $this->assertLessThan(1e-7,$backend->scalar($backend->amax($diff)));
        $this->assertEquals($o->toArray(),$ocl->toArray());
    }


    public function testDuplicate()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $backendCL = $this->newBackendCL($mo);
        $la = $mo->la();
        $lacl = $mo->laAccelerated('clblast');

        $a = $backend->glorot_normal([32,2]);
        $acl = $backendCL->zerosLike($a);
        $backendCL->copy($backendCL->array($a),$acl);

        $o = $la->duplicate($a);
        $ocl = $lacl->duplicate($acl);
        $this->assertEquals($o->toArray(),$ocl->toArray());
    }

    public function testSigmoidFunction()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $backendCL = $this->newBackendCL($mo);
        $nncl = new NeuralNetworks($mo,$backendCL);

        $sigmoid = FunctionFactory::factory($backend,'sigmoid');
        $sigmoidcl = FunctionFactory::factory($backendCL,'sigmoid');

        $a = $backend->glorot_normal([32,2]);
        $acl = $backendCL->zerosLike($a);
        $backendCL->copy($backendCL->array($a),$acl);

        $states = new \stdClass();
        $statesCL = new \stdClass();
        $o = $sigmoid->forward($states,$a,true);
        $ocl = $sigmoidcl->forward($statesCL,$acl,true);
        $diff = $backend->sub($o,$backendCL->ndarray($ocl));
        $this->assertLessThan(1e-7,$backend->scalar($backend->amax($diff)));
        //$this->assertEquals($o->toArray(),$ocl->toArray());
    }

    public function testScal()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $backendCL = $this->newBackendCL($mo);
        $la = $mo->laRawMode();
        $lacl = $mo->laAccelerated('clblast');

        $a = $backend->glorot_normal([32,2]);
        $acl = $backendCL->zerosLike($a);
        $backendCL->copy($backendCL->array($a),$acl);

        $o = $la->scal(-1.0,$a);
        $ocl = $lacl->scal(-1.0,$acl);
        $this->assertEquals($o->toArray(),$ocl->toArray());
    }

    public function testexp()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $backendCL = $this->newBackendCL($mo);
        $la = $mo->laRawMode();
        $lacl = $mo->laAccelerated('clblast');

        $a = $backend->glorot_normal([32,2]);
        $acl = $backendCL->zerosLike($a);
        $backendCL->copy($backendCL->array($a),$acl);

        $o = $la->exp($a);
        $ocl = $lacl->exp($acl);
        $diff = $backend->sub($o,$backendCL->ndarray($ocl));
        // ******** CAUTION: diff=1e-6 *********
        $this->assertLessThan(1e-6,$backend->scalar($backend->amax($diff)));
        //echo $mo->toString($backend->sub($o,$backendCL->ndarray($ocl)));
        //$this->assertEquals($o->toArray(),$ocl->toArray());
    }

}
