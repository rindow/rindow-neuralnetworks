<?php
namespace Rindow\NeuralNetworks\Support\HDA;

use RuntimeException;
use PDO;
use Iterator;

class HDASqliteIterator implements Iterator
{
    protected $key;
    protected $value;
    protected $stat;
    protected $eof = false;
    protected $fetched = false;

    public function __construct(PDO $pdo, string $table, string $ancestor)
    {
        $this->stat = $pdo->prepare("SELECT * FROM ".$table." WHERE ancestor = :ancestor");
        if(!$this->stat->execute([':ancestor'=>$ancestor]))
            throw new RuntimeException('query error in '.$ancestor);
    }

    protected function fetch()
    {
        if($this->fetched)
            return;
        if($this->eof)
            return;
        $row = $this->stat->fetch(PDO::FETCH_ASSOC);
        $this->fetched = true;
        if(!$row) {
            $this->eof = true;
            $this->value = null;
            $this->key = null;
            return;
        }
        $this->value = $row['value'];
        $this->key = $row['key'];
    }

    public function rewind() : void
    {
        $this->fetch();
    }

    public function valid() : bool
    {
        $this->fetch();
        return !$this->eof;
    }

    public function current()
    {
        $this->fetch();
        return $this->value;
    }

    public function key()
    {
        $this->fetch();
        return $this->key;
    }

    public function next() : void
    {
        $this->fetched = false;
        $this->fetch();
    }

    public function __destruct()
    {
        $this->stat->closeCursor();
    }
}
