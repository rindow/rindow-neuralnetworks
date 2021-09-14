<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use LogicException;
use Throwable;
use Rindow\NeuralNetworks\Support\Control\Context;
use Rindow\NeuralNetworks\Layer\LayerBase;

class BuildContext implements Context
{
    static public $build = false;
    static public $ctx;
    protected $layers = [];

    static public function add($layer) : void
    {
        if(!self::$build) {
            throw new InvalidArgumentException('Not in the build context');
        }
        self::$ctx->addEntry($layer);
    }

    public function enter() : void
    {
        if(self::$build) {
            throw new InvalidArgumentException('Already in the build context');
        }
        self::$ctx = $this;
        self::$build = true;
    }

    public function exit(Throwable $e=null) : bool
    {
        self::$build = false;
        self::$ctx = null;
        return false;
    }

    public function addEntry($layer) : void
    {
        if(!self::$build) {
            throw new InvalidArgumentException('Not in the build context');
        }
        $this->layers[] = $layer;
    }

    public function getList() : array
    {
        return $this->layers;
    }
}
